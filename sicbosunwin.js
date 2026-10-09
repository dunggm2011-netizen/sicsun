// ============================================================================
// SICBO SUNWIN AI SERVER v5.0 - FULL INTEGRATION
// Kết hợp: 10 thuật toán cũ + 8 cầu + AI tự học + Adaptive Weight
// ============================================================================

import fastify from "fastify";
import cors from "@fastify/cors";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import fetch from "node-fetch";
import fs from "fs";

// ============================================================================
// CẤU HÌNH
// ============================================================================
const port = process.env.PORT || 3000;
const api_url = process.env.API_URL ||
    "https://api.wsktnus8.net/v2/history/getLastResult?gameId=ktrng_3979&size=100&tableId=39791215743193&curPage=1";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// File lưu trạng thái AI học (persist qua restart)
const AI_STATE_FILE = path.join(__dirname, "ai_state.json");

// ============================================================================
// GLOBAL STATE
// ============================================================================
let txh_history = [];
let current_session_id = null;
let fetch_interval = null;
let prediction_memory = { T: [], X: [] };
let last_prediction = null;

// ============================================================================
// AI STATE - TỰ HỌC
// ============================================================================
const aiState = {
    // Trọng số từng thuật toán (cập nhật theo độ chính xác)
    algoWeights: {},

    // Trọng số từng loại cầu
    roadWeights: {},

    // Thống kê theo thuật toán
    algoStats: {},

    // Thống kê theo cầu
    roadStats: {},

    // Ma trận chuyển đổi cầu (cầu A → cầu B tiếp theo)
    roadTransitions: {},

    // Trọng số theo pattern 5 phiên
    patternWeights: {},

    // Thống kê chung
    total: 0,
    correct: 0,

    // Độ chính xác theo từng bucket confidence
    confidenceBuckets: {
        very_high: { total: 0, correct: 0 }, // > 75%
        high: { total: 0, correct: 0 },      // 65-75%
        medium: { total: 0, correct: 0 },    // 55-65%
        low: { total: 0, correct: 0 },       // < 55%
    },
};

const prediction_stats = {
    total: 0,
    correct: 0,
    log: [],
};

// ============================================================================
// LOAD / SAVE AI STATE
// ============================================================================
function loadAiState() {
    try {
        if (fs.existsSync(AI_STATE_FILE)) {
            const data = JSON.parse(fs.readFileSync(AI_STATE_FILE, "utf8"));
            Object.assign(aiState, data);
            console.log(`[AI] Loaded state: ${aiState.total} predictions, acc ${getOverallAccuracy()}%`);
        }
    } catch (e) {
        console.warn("[AI] Không load được state:", e.message);
    }
}

function saveAiState() {
    try {
        fs.writeFileSync(AI_STATE_FILE, JSON.stringify(aiState, null, 2));
    } catch (e) {
        console.warn("[AI] Không save được state:", e.message);
    }
}

function getOverallAccuracy() {
    if (aiState.total === 0) return 0;
    return (aiState.correct / aiState.total * 100).toFixed(1);
}

// ============================================================================
// PHẦN 1: PARSE DATA
// ============================================================================
function parse_lines(data) {
    if (!data || !data.data || !Array.isArray(data.data.resultList)) return [];

    const sorted = data.data.resultList.sort((a, b) => {
        return parseInt(b.gameNum.slice(1)) - parseInt(a.gameNum.slice(1));
    });

    return sorted.map(item => {
        const total = item.score;
        let tx, result_text;

        if (total >= 4 && total <= 10) { tx = 'X'; result_text = "XIU"; }
        else if (total >= 11 && total <= 17) { tx = 'T'; result_text = "TAI"; }
        else if (total === 3 || total === 18) { tx = 'B'; result_text = "BAO"; }
        else { tx = 'N'; result_text = "UNKNOWN"; }

        const dice = Array.isArray(item.facesList)
            ? item.facesList
            : (typeof item.keyR === 'string' ? item.keyR.split('-').map(Number) : [0, 0, 0]);

        return {
            session: parseInt(item.gameNum.slice(1)),
            dice, total, result: result_text, tx,
        };
    }).sort((a, b) => a.session - b.session);
}

// ============================================================================
// PHẦN 2: 10 THUẬT TOÁN CŨ (từ bản gốc)
// ============================================================================

function last_n(arr, n) { return arr.slice(Math.max(0, arr.length - n)); }
function sum(nums) { return nums.reduce((a, b) => a + b, 0); }
function avg(nums) { return nums.length ? sum(nums) / nums.length : 0; }

function entropy(arr) {
    if (!arr.length) return 0;
    const freq = arr.reduce((a, v) => { a[v] = (a[v] || 0) + 1; return a; }, {});
    const n = arr.length;
    let e = 0;
    for (let k in freq) { const p = freq[k] / n; e -= p * Math.log2(p); }
    return e;
}

function similarity(a, b) {
    if (a.length !== b.length) return 0;
    let m = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) m++;
    return m / a.length;
}

function extract_features(history) {
    const filtered = history.filter(h => h.tx !== 'B');
    const tx = filtered.map(h => h.tx);
    const totals = filtered.map(h => h.total);

    const features = {
        tx, totals,
        freq: tx.reduce((a, v) => { a[v] = (a[v] || 0) + 1; return a; }, {})
    };

    let runs = [], cur = tx[0], len = 1;
    for (let i = 1; i < tx.length; i++) {
        if (tx[i] === cur) len++;
        else { runs.push({ val: cur, len }); cur = tx[i]; len = 1; }
    }
    if (tx.length) runs.push({ val: cur, len });
    features.runs = runs;
    features.max_run = runs.reduce((m, r) => Math.max(m, r.len), 0) || 0;
    features.mean_total = avg(totals);
    features.std_total = Math.sqrt(avg(totals.map(t => Math.pow(t - features.mean_total, 2))));
    features.entropy = entropy(tx);

    return features;
}

// --- ALGO 1: Frequency Rebalance ---
function algo5_freq_rebalance(history) {
    if (history.length < 15) return null;
    const features = extract_features(history);
    const tx = features.tx;
    const total_t = (features.freq['T'] || 0);
    const total_x = (features.freq['X'] || 0);
    if (tx.length === 0) return null;

    const recent_30 = tx.slice(-30);
    const recent_t = recent_30.filter(x => x === 'T').length;
    const recent_x = recent_30.filter(x => x === 'X').length;

    const recent_10 = tx.slice(-10);
    const recent_10_t = recent_10.filter(x => x === 'T').length;
    const recent_10_x = recent_10.filter(x => x === 'X').length;

    if (recent_10_t >= 7) return 'X';
    if (recent_10_x >= 7) return 'T';
    if (total_t > total_x + 8 && recent_t > recent_x + 2) return 'X';
    if (total_x > total_t + 8 && recent_x > recent_t + 2) return 'T';
    if (recent_10_t > recent_10_x + 3) return 'X';
    if (recent_10_x > recent_10_t + 3) return 'T';
    return null;
}

// --- ALGO 2: Markov Enhanced ---
function algoa_markov(history) {
    const tx = extract_features(history).tx;
    if (tx.length < 20) return null;

    let best_pred = null, best_confidence = 0;
    for (let order = 2; order <= 5; order++) {
        if (tx.length < order + 5) continue;
        const transitions = {};
        for (let i = 0; i <= tx.length - order - 1; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            transitions[key] = transitions[key] || { t: 0, x: 0 };
            transitions[key][next.toLowerCase()]++;
        }
        const last_key = tx.slice(-order).join('');
        const counts = transitions[last_key];
        if (counts && (counts.t > 0 || counts.x > 0)) {
            const total = counts.t + counts.x;
            const confidence = Math.abs(counts.t - counts.x) / total;
            if (confidence > best_confidence && confidence > 0.6) {
                best_confidence = confidence;
                best_pred = counts.t > counts.x ? 'T' : 'X';
            }
        }
    }
    return best_pred;
}

// --- ALGO 3: N-gram Advanced ---
function algob_ngram(history) {
    const tx = extract_features(history).tx;
    for (let k = 3; k <= 6; k++) {
        if (tx.length < k + 10) continue;
        const last_gram = tx.slice(-k).join('');
        let counts = { t: 0, x: 0 }, total_matches = 0;
        for (let i = 0; i <= tx.length - k - 1; i++) {
            if (tx.slice(i, i + k).join('') === last_gram) {
                counts[tx[i + k].toLowerCase()]++;
                total_matches++;
            }
        }
        if (total_matches >= 3) {
            const ratio = Math.abs(counts.t - counts.x) / total_matches;
            if (ratio >= 0.6) return counts.t > counts.x ? 'T' : 'X';
        }
    }
    return null;
}

// --- ALGO 4: Neo Pattern ---
function algos_neo_pattern(history) {
    const tx = extract_features(history).tx;
    const len = tx.length;
    if (len < 40) return null;

    let best_pred = null, max_confidence = 0;
    for (let pat_len of [4, 5, 6]) {
        if (len < pat_len * 2) continue;
        const target = tx.slice(-pat_len);
        let counts = { t: 0, x: 0 };

        for (let i = 0; i <= len - pat_len - 1; i++) {
            const hist = tx.slice(i, i + pat_len);
            let match = 0;
            for (let j = 0; j < pat_len; j++) if (hist[j] === target[j]) match++;
            if (match / pat_len >= 0.8) counts[tx[i + pat_len].toLowerCase()]++;
        }

        const total = counts.t + counts.x;
        if (total >= 5) {
            const conf = Math.abs(counts.t - counts.x) / total;
            if (conf > max_confidence && conf > 0.6) {
                max_confidence = conf;
                best_pred = counts.t > counts.x ? 'T' : 'X';
            }
        }
    }
    return best_pred;
}

// --- ALGO 5: Super Deep Analysis ---
function algof_super_deep_analysis(history) {
    if (history.length < 80) return null;
    const features = extract_features(history);
    const tx = features.tx, totals = features.totals, runs = features.runs;

    const recent_50 = tx.slice(-50);
    const recent_50_t = recent_50.filter(x => x === 'T').length;
    const recent_50_x = recent_50.filter(x => x === 'X').length;
    const recent_totals = totals.slice(-30);
    const avg_recent = avg(recent_totals);

    let pattern_pred = null;
    const last_10 = tx.slice(-10);
    const patterns = {
        'TTXTT': 'X', 'TXTXT': 'X', 'XXTXX': 'T',
        'XTXTX': 'T', 'TTTTX': 'X', 'XXXXT': 'T'
    };
    const last_5 = last_10.slice(-5).join('');
    if (patterns[last_5]) pattern_pred = patterns[last_5];

    let run_pred = null;
    if (runs.length >= 2) {
        const last_run = runs[runs.length - 1];
        if (last_run.len >= 4) run_pred = last_run.val === 'T' ? 'X' : 'T';
        else if (last_run.len === 1) {
            const last_3 = runs.slice(-3);
            if (last_3.length === 3 && last_3.every(r => r.len === 1)) {
                run_pred = last_run.val === 'T' ? 'X' : 'T';
            }
        }
    }

    const votes = [];
    if (Math.abs(recent_50_t - recent_50_x) > 10) votes.push(recent_50_t > recent_50_x ? 'T' : 'X');
    if (avg_recent > 13.0) votes.push('X');
    else if (avg_recent < 8.5) votes.push('T');
    if (pattern_pred) votes.push(pattern_pred);
    if (run_pred) votes.push(run_pred);

    if (votes.length === 0) return null;
    const t_votes = votes.filter(x => x === 'T').length;
    const x_votes = votes.filter(x => x === 'X').length;
    if (t_votes > x_votes && t_votes >= 2) return 'T';
    if (x_votes > t_votes && x_votes >= 2) return 'X';
    return null;
}

// --- ALGO 6: Transformer Pro ---
function algoe_transformer(history) {
    const tx = extract_features(history).tx;
    const len = tx.length;
    if (len < 60) return null;

    const window_size = 10;
    const target_seq = tx.slice(-window_size).join('');
    let t_weight = 0, x_weight = 0, total_sim = 0;

    for (let i = 0; i <= len - window_size - 1; i++) {
        const hist_seq = tx.slice(i, i + window_size).join('');
        const sim = similarity(hist_seq, target_seq);
        if (sim > 0.7) {
            const weight = sim * (1 / (len - i + 10));
            if (tx[i + window_size] === 'T') t_weight += weight;
            else x_weight += weight;
            total_sim += sim;
        }
    }

    if (total_sim > 2.0) {
        const total = t_weight + x_weight;
        if (total > 0 && Math.abs(t_weight - x_weight) / total > 0.6) {
            return t_weight > x_weight ? 'T' : 'X';
        }
    }
    return null;
}

// --- ALGO 7: Super Bridge ---
function algog_super_bridge_predictor(history) {
    const features = extract_features(history);
    const runs = features.runs, tx = features.tx;
    if (runs.length < 4) return null;

    const last_5 = runs.slice(-5);
    const recent_tx = tx.slice(-15);
    if (last_5.length < 5) return null;

    const last_run = last_5[4];
    const second_last = last_5[3];

    if (last_run.len >= 4) return last_run.val === 'T' ? 'X' : 'T';

    if (last_run.len === 1 && second_last.len === 1) {
        let alt = 0;
        for (let i = runs.length - 1; i >= Math.max(0, runs.length - 6); i--) {
            if (runs[i].len === 1) alt++;
        }
        if (alt >= 4) return last_run.val === 'T' ? 'X' : 'T';
    }

    const t_recent = recent_tx.filter(x => x === 'T').length;
    const x_recent = recent_tx.filter(x => x === 'X').length;
    if (t_recent > x_recent + 4) return 'X';
    if (x_recent > t_recent + 4) return 'T';
    return null;
}

// --- ALGO 8: Adaptive Markov ---
function algo_h_adaptive_markov(history) {
    const tx = extract_features(history).tx;
    if (tx.length < 25) return null;

    let best_pred = null, best_conf = 0;
    const max_order = Math.min(4, Math.floor(tx.length / 10));

    for (let order = 2; order <= max_order; order++) {
        if (tx.length < order + 5) continue;
        const trans = {};
        for (let i = 0; i <= tx.length - order - 1; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            trans[key] = trans[key] || { t: 0, x: 0 };
            trans[key][next.toLowerCase()]++;
        }
        const last_key = tx.slice(-order).join('');
        const counts = trans[last_key];
        if (counts && counts.t + counts.x >= 2) {
            const total = counts.t + counts.x;
            const conf = Math.abs(counts.t - counts.x) / total;
            if (conf > best_conf) {
                best_conf = conf;
                best_pred = counts.t > counts.x ? 'T' : 'X';
            }
        }
    }
    if (best_conf > 0.7) return best_pred;
    return null;
}

// --- ALGO 9: Neural Pattern ---
function algoi_neural_pattern(history) {
    const features = extract_features(history);
    const tx = features.tx, totals = features.totals;
    if (tx.length < 50) return null;

    const recent_20 = tx.slice(-20);
    const t_count_20 = recent_20.filter(x => x === 'T').length;
    const ratio_t_20 = t_count_20 / 20;
    const recent_totals_20 = totals.slice(-20);
    const avg_total_20 = avg(recent_totals_20);

    const last_5 = tx.slice(-5);
    const last_5_pattern = last_5.join('');

    const first_10_avg = totals.length >= 20 ? avg(totals.slice(-20, -10)) : 0;
    const last_10_avg = totals.length >= 10 ? avg(totals.slice(-10)) : 0;
    const trend = last_10_avg - first_10_avg;

    let t_score = 0, x_score = 0;

    if (ratio_t_20 > 0.7) x_score += 0.35;
    else if (ratio_t_20 < 0.3) t_score += 0.35;

    if (avg_total_20 > 12.5) x_score += 0.25;
    else if (avg_total_20 < 8.5) t_score += 0.25;

    const patterns = {
        'TTXTT': { t: 0.2, x: 0.8 }, 'TXTXT': { t: 0.3, x: 0.7 },
        'XXTXX': { t: 0.8, x: 0.2 }, 'XTXTX': { t: 0.7, x: 0.3 },
        'TTTTX': { t: 0.1, x: 0.9 }, 'XXXXT': { t: 0.9, x: 0.1 }
    };
    if (patterns[last_5_pattern]) {
        t_score += patterns[last_5_pattern].t * 0.2;
        x_score += patterns[last_5_pattern].x * 0.2;
    }

    if (trend > 2.5) x_score += 0.15;
    else if (trend < -2.5) t_score += 0.15;

    if (Math.abs(t_score - x_score) > 0.25) {
        return t_score > x_score ? 'T' : 'X';
    }
    return null;
}

// --- ALGO 10: Quantum Predictor ---
function algoj_quantum_predictor(history) {
    const tx = extract_features(history).tx;
    if (tx.length < 40) return null;

    let sup_t = 0, sup_x = 0;

    const recent_30 = tx.slice(-30);
    const t_30 = recent_30.filter(x => x === 'T').length;
    const x_30 = recent_30.filter(x => x === 'X').length;
    if (t_30 > x_30 + 5) sup_x += 0.3;
    else if (x_30 > t_30 + 5) sup_t += 0.3;

    const last_8 = tx.slice(-8);
    const pattern_weights = { t: 0, x: 0 };
    for (let i = 0; i <= tx.length - 9; i++) {
        if (tx.slice(i, i + 8).join('') === last_8.join('')) {
            pattern_weights[tx[i + 8].toLowerCase()]++;
        }
    }
    if (pattern_weights.t + pattern_weights.x >= 3) {
        const ratio = Math.abs(pattern_weights.t - pattern_weights.x) / (pattern_weights.t + pattern_weights.x);
        if (ratio > 0.6) {
            if (pattern_weights.t > pattern_weights.x) sup_t += 0.25;
            else sup_x += 0.25;
        }
    }

    const entropy_val = entropy(tx.slice(-20));
    if (entropy_val > 0.9) {
        if (tx[tx.length - 1] === 'T') sup_x += 0.2;
        else sup_t += 0.2;
    }

    const total = sup_t + sup_x;
    if (total > 0.5) {
        const conf = Math.abs(sup_t - sup_x) / total;
        if (conf > 0.6) return sup_t > sup_x ? 'T' : 'X';
    }
    return null;
}

// --- DANH SÁCH 10 THUẬT TOÁN ---
const all_algs = [
    { id: 'algo5_freq_rebalance', fn: algo5_freq_rebalance },
    { id: 'a_markov', fn: algoa_markov },
    { id: 'b_ngram', fn: algob_ngram },
    { id: 's_neo_pattern', fn: algos_neo_pattern },
    { id: 'f_super_deep_analysis', fn: algof_super_deep_analysis },
    { id: 'e_transformer', fn: algoe_transformer },
    { id: 'g_super_bridge_predictor', fn: algog_super_bridge_predictor },
    { id: 'h_adaptive_markov', fn: algo_h_adaptive_markov },
    { id: 'i_neural_pattern', fn: algoi_neural_pattern },
    { id: 'j_quantum_predictor', fn: algoj_quantum_predictor },
];

// ============================================================================
// PHẦN 3: 8 LOẠI CẦU
// ============================================================================

function detectRoadPattern(history) {
    const seq = history.filter(h => h.tx !== 'B').map(h => h.tx);
    if (seq.length < 4) return null;

    const patterns = [
        detectAlternating(seq),
        detectBlock22(seq),
        detectAsymmetric(seq, 3, 2),
        detectAsymmetric(seq, 3, 1),
        detectAsymmetric(seq, 2, 1),
        detectDouble(seq),
        detectLongRun(seq),
        detectTrend(seq),
    ].filter(Boolean);

    if (patterns.length === 0) return null;
    patterns.sort((a, b) => b.strength - a.strength);
    return patterns[0];
}

function buildBlocks(seq) {
    const blocks = [];
    let cur = { val: seq[seq.length - 1], len: 1 };
    for (let i = seq.length - 2; i >= 0; i--) {
        if (seq[i] === cur.val) cur.len++;
        else { blocks.unshift(cur); cur = { val: seq[i], len: 1 }; }
    }
    blocks.unshift(cur);
    return blocks;
}

function detectAlternating(seq) {
    if (seq.length < 5) return null;
    let count = 0;
    for (let i = seq.length - 1; i > 0; i--) {
        if (seq[i] !== seq[i - 1]) count++;
        else break;
    }
    if (count >= 4) {
        const last = seq[seq.length - 1];
        return {
            name: '1-1', type: 'theo',
            prediction: last === 'T' ? 'X' : 'T',
            strength: Math.min(0.85, 0.4 + count * 0.08),
            desc: `Theo cầu 1-1 (${count} phiên)`,
        };
    }
    return null;
}

function detectBlock22(seq) {
    if (seq.length < 8) return null;
    const blocks = buildBlocks(seq);
    if (blocks.length < 3) return null;

    const last = blocks[blocks.length - 1];
    const prev = blocks[blocks.length - 2];
    const prev2 = blocks[blocks.length - 3];

    if (prev.len === 2 && prev2.len === 2) {
        if (last.len < 2) {
            return { name: '2-2', type: 'theo', prediction: last.val, strength: 0.72, desc: `Theo cầu 2-2 (${last.len}/2)` };
        } else {
            return { name: '2-2', type: 'be', prediction: last.val === 'T' ? 'X' : 'T', strength: 0.78, desc: 'Bẻ cầu 2-2' };
        }
    }
    return null;
}

function detectAsymmetric(seq, lenA, lenB) {
    if (seq.length < lenA + lenB + 2) return null;
    const blocks = buildBlocks(seq);
    if (blocks.length < 3) return null;

    const [b3, b2, b1] = blocks.slice(-3);
    const matchABA = b3.len === lenA && b2.len === lenB && b1.len === lenA;
    const matchBAB = b3.len === lenB && b2.len === lenA && b1.len === lenB;

    if (matchABA || matchBAB) {
        const last = blocks[blocks.length - 1];
        const expected = matchABA ? lenA : lenB;
        if (last.len < expected) {
            return { name: `${lenA}-${lenB}`, type: 'theo', prediction: last.val, strength: 0.75, desc: `Theo cầu ${lenA}-${lenB}` };
        } else {
            return { name: `${lenA}-${lenB}`, type: 'be', prediction: last.val === 'T' ? 'X' : 'T', strength: 0.80, desc: `Bẻ cầu ${lenA}-${lenB}` };
        }
    }
    return null;
}

function detectDouble(seq) {
    if (seq.length < 4) return null;
    const last = seq[seq.length - 1];
    const prev = seq[seq.length - 2];
    if (last !== prev) return null;

    let count = 2;
    for (let i = seq.length - 3; i >= 0; i--) {
        if (seq[i] === last) count++;
        else break;
    }
    if (count === 2) {
        return { name: `Đôi ${last}${last}`, type: 'be', prediction: last === 'T' ? 'X' : 'T', strength: 0.68, desc: `Bẻ cầu Đôi ${last}${last}` };
    }
    return null;
}

function detectLongRun(seq) {
    if (seq.length < 4) return null;
    const last = seq[seq.length - 1];
    let count = 1;
    for (let i = seq.length - 2; i >= 0; i--) {
        if (seq[i] === last) count++;
        else break;
    }
    if (count >= 3) {
        const shouldBreak = count >= 4;
        return {
            name: 'Thang Lời',
            type: shouldBreak ? 'be' : 'theo',
            prediction: shouldBreak ? (last === 'T' ? 'X' : 'T') : last,
            strength: Math.min(0.88, 0.5 + count * 0.08),
            desc: shouldBreak ? `Bẻ cầu Thang Lời ${count} phiên` : `Theo cầu Thang Lời ${count} phiên`,
        };
    }
    return null;
}

function detectTrend(seq) {
    if (seq.length < 15) return null;
    const recent = seq.slice(-15);
    const t = recent.filter(x => x === 'T').length;
    const x = recent.filter(x => x === 'X').length;
    if (Math.abs(t - x) >= 6) {
        return { name: 'Nghiêng', type: 'be', prediction: t > x ? 'X' : 'T', strength: 0.62, desc: `Bẻ cầu Nghiêng ${t > x ? 'Tài' : 'Xỉu'}` };
    }
    return null;
}

// ============================================================================
// PHẦN 4: AI TỰ HỌC - ADAPTIVE WEIGHT
// ============================================================================

/**
 * Khởi tạo trọng số ban đầu
 */
function initAiWeights() {
    // Trọng số cho 10 thuật toán (mặc định = 1.0)
    for (const alg of all_algs) {
        if (!aiState.algoWeights[alg.id]) {
            aiState.algoWeights[alg.id] = 1.0;
            aiState.algoStats[alg.id] = { total: 0, correct: 0, recentCorrect: 0, recentTotal: 0 };
        }
    }

    // Trọng số cho các loại cầu
    const roadNames = ['1-1', '2-2', '3-2', '3-1', '2-1', 'Đôi TT', 'Đôi XX', 'Thang Lời', 'Nghiêng'];
    for (const name of roadNames) {
        if (!aiState.roadWeights[name]) {
            aiState.roadWeights[name] = 1.0;
            aiState.roadStats[name] = { total: 0, correct: 0 };
        }
    }
}

/**
 * Cập nhật trọng số sau mỗi kết quả thực tế
 * @param {Object} actual - record thực tế {session, tx, total}
 * @param {Object} lastPred - dự đoán trước đó
 * @param {Array} historyPrefix - lịch sử trước khi có actual
 */
function updateAiWeights(actual, lastPred, historyPrefix) {
    if (actual.tx === 'B') return; // bỏ qua bão

    aiState.total++;
    const isCorrect = lastPred.prediction === actual.tx;
    if (isCorrect) aiState.correct++;

    // 1. Cập nhật trọng số từng thuật toán
    for (const alg of all_algs) {
        const pred = alg.fn(historyPrefix);
        if (!pred) continue;

        const stats = aiState.algoStats[alg.id];
        stats.total++;
        stats.recentTotal++;
        if (pred === actual.tx) {
            stats.correct++;
            stats.recentCorrect++;
        }

        // Reset cửa sổ recent mỗi 100 mẫu
        if (stats.recentTotal >= 100) {
            stats.recentTotal = Math.floor(stats.recentTotal / 2);
            stats.recentCorrect = Math.floor(stats.recentCorrect / 2);
        }

        // Cập nhật weight theo EMA (Exponential Moving Average)
        const overallAcc = stats.total > 0 ? stats.correct / stats.total : 0.5;
        const recentAcc = stats.recentTotal > 0 ? stats.recentCorrect / stats.recentTotal : 0.5;

        // Trọng số kết hợp: 40% overall + 60% recent
        const combinedAcc = overallAcc * 0.4 + recentAcc * 0.6;

        // Áp dụng power để khuếch đại khác biệt
        const targetWeight = Math.pow(combinedAcc, 2);

        // Smooth: weight mới = 0.85 * cũ + 0.15 * target
        const oldWeight = aiState.algoWeights[alg.id] || 1.0;
        aiState.algoWeights[alg.id] = Math.max(0.1, oldWeight * 0.85 + targetWeight * 0.15);
    }

    // 2. Cập nhật trọng số theo cầu
    if (lastPred.roadPattern) {
        const roadName = normalizeRoadName(lastPred.roadPattern.name);
        const stats = aiState.roadStats[roadName] = aiState.roadStats[roadName] || { total: 0, correct: 0 };
        stats.total++;
        if (isCorrect) stats.correct++;

        const acc = stats.total > 0 ? stats.correct / stats.total : 0.5;
        const oldW = aiState.roadWeights[roadName] || 1.0;
        aiState.roadWeights[roadName] = Math.max(0.3, oldW * 0.9 + acc * 0.1);
    }

    // 3. Cập nhật ma trận chuyển đổi cầu
    // Ví dụ: sau cầu 2-2, cầu tiếp theo thường là gì?
    if (lastPred.roadPattern && historyPrefix.length > 0) {
        const prevRoad = detectRoadPattern(historyPrefix.slice(0, -1));
        if (prevRoad) {
            const from = normalizeRoadName(prevRoad.name);
            const to = normalizeRoadName(lastPred.roadPattern.name);
            aiState.roadTransitions[from] = aiState.roadTransitions[from] || {};
            aiState.roadTransitions[from][to] = (aiState.roadTransitions[from][to] || 0) + 1;
        }
    }

    // 4. Cập nhật bucket confidence
    const conf = lastPred.confidence;
    const bucket = conf > 0.75 ? 'very_high' : conf > 0.65 ? 'high' : conf > 0.55 ? 'medium' : 'low';
    aiState.confidenceBuckets[bucket].total++;
    if (isCorrect) aiState.confidenceBuckets[bucket].correct++;

    // 5. Cập nhật pattern weights (5 phiên cuối)
    const seq = historyPrefix.filter(h => h.tx !== 'B').map(h => h.tx);
    if (seq.length >= 5) {
        const pattern = seq.slice(-5).join('');
        aiState.patternWeights[pattern] = aiState.patternWeights[pattern] || { T: 0, X: 0, total: 0 };
        aiState.patternWeights[pattern].total++;
        if (actual.tx === 'T') aiState.patternWeights[pattern].T++;
        else aiState.patternWeights[pattern].X++;
    }

    // Lưu state định kỳ (mỗi 10 dự đoán)
    if (aiState.total % 10 === 0) saveAiState();
}

function normalizeRoadName(name) {
    // Chuẩn hóa tên cầu để thống kê
    if (name.startsWith('Đôi')) {
        if (name.includes('TT')) return 'Đôi TT';
        if (name.includes('XX')) return 'Đôi XX';
        return 'Đôi';
    }
    return name;
}

/**
 * Lấy trọng số thuật toán
 */
function getAlgoWeight(algoId) {
    return aiState.algoWeights[algoId] || 1.0;
}

/**
 * Lấy trọng số cầu
 */
function getRoadWeight(roadName) {
    return aiState.roadWeights[normalizeRoadName(roadName)] || 1.0;
}

/**
 * Dự đoán từ pattern 5 phiên (AI học)
 */
function aiPatternPredict(history) {
    const seq = history.filter(h => h.tx !== 'B').map(h => h.tx);
    if (seq.length < 6) return null;

    const pattern = seq.slice(-5).join('');
    const data = aiState.patternWeights[pattern];
    if (!data || data.total < 3) return null;

    const pT = data.T / data.total;
    const pX = data.X / data.total;

    if (Math.abs(pT - pX) > 0.25) {
        return {
            prediction: pT > pX ? 'T' : 'X',
            confidence: Math.max(pT, pX),
            source: 'ai_pattern',
        };
    }
    return null;
}

/**
 * Dự đoán từ ma trận chuyển đổi cầu
 */
function aiRoadTransitionPredict(history) {
    const seq = history.filter(h => h.tx !== 'B').map(h => h.tx);
    if (seq.length < 8) return null;

    // Cầu hiện tại
    const currentRoad = detectRoadPattern(history);
    if (!currentRoad) return null;

    const from = normalizeRoadName(currentRoad.name);
    const transitions = aiState.roadTransitions[from];
    if (!transitions) return null;

    // Tìm cầu kế tiếp có xác suất cao nhất
    const entries = Object.entries(transitions).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) return null;

    const [nextRoad, count] = entries[0];
    const totalTrans = Object.values(transitions).reduce((a, b) => a + b, 0);
    const prob = count / totalTrans;

    if (prob < 0.4) return null;

    // Suy đoán hướng dựa trên cầu kế tiếp
    let prediction = null;
    if (nextRoad.includes('1-1')) prediction = currentRoad.prediction;
    else if (nextRoad.includes('2-2') || nextRoad.includes('3-2') || nextRoad.includes('Thang')) {
        prediction = currentRoad.prediction;
    } else if (nextRoad.includes('Đôi') || nextRoad.includes('Nghiêng')) {
        prediction = currentRoad.prediction === 'T' ? 'X' : 'T';
    }

    if (prediction) {
        return {
            prediction,
            confidence: prob,
            source: `ai_transition→${nextRoad}`,
        };
    }
    return null;
}

// ============================================================================
// PHẦN 5: ENSEMBLE VOTING KẾT HỢP AI
// ============================================================================

function ensemblePredict(history, roadPattern) {
    const seq = history.filter(h => h.tx !== 'B').map(h => h.tx);

    // 1. Vote từ 10 thuật toán cũ (có weight từ AI)
    const algoVotes = { T: 0, X: 0, none: 0 };
    const algoDetails = {};

    for (const alg of all_algs) {
        const pred = alg.fn(history);
        if (!pred) {
            algoVotes.none++;
            algoDetails[alg.id] = null;
            continue;
        }

        const weight = getAlgoWeight(alg.id);
        algoVotes[pred] += weight;
        algoDetails[alg.id] = { pred, weight };
    }

    let tScore = algoVotes.T;
    let xScore = algoVotes.X;

    // 2. Vote từ cầu (có weight từ AI)
    let roadVote = null;
    if (roadPattern) {
        const roadW = getRoadWeight(roadPattern.name) * 2.5;
        const roadScore = roadW * roadPattern.strength;

        if (roadPattern.prediction === 'T') tScore += roadScore;
        else xScore += roadScore;

        roadVote = {
            name: roadPattern.name,
            type: roadPattern.type,
            pred: roadPattern.prediction,
            weight: roadScore,
        };
    }

    // 3. Vote từ AI pattern (5 phiên)
    const aiPat = aiPatternPredict(history);
    if (aiPat) {
        const w = aiPat.confidence * 1.5;
        if (aiPat.prediction === 'T') tScore += w;
        else xScore += w;
    }

    // 4. Vote từ AI transition
    const aiTrans = aiRoadTransitionPredict(history);
    if (aiTrans) {
        const w = aiTrans.confidence * 1.5;
        if (aiTrans.prediction === 'T') tScore += w;
        else xScore += w;
    }

    // 5. Kết luận
    const total = tScore + xScore;
    if (total === 0) {
        return {
            prediction: 'T',
            confidence: 0.5,
            algoVotes,
            algoDetails,
            roadVote,
            aiPatternVote: aiPat,
            aiTransitionVote: aiTrans,
            reason: 'Fallback cân bằng',
        };
    }

    const prediction = tScore > xScore ? 'T' : 'X';
    const confidence = Math.max(tScore, xScore) / total;

    let reason = '';
    if (roadPattern) reason = roadPattern.desc;
    else if (confidence > 0.65) reason = `Ensemble AI | ${(confidence * 100).toFixed(0)}%`;
    else reason = `Ensemble AI | Fallback theo nhịp`;

    return {
        prediction,
        confidence,
        algoVotes,
        algoDetails,
        roadVote,
        aiPatternVote: aiPat,
        aiTransitionVote: aiTrans,
        reason,
    };
}

// ============================================================================
// PHẦN 6: DỰ ĐOÁN VỊ
// ============================================================================

function predictPositions(history, txConstraint) {
    const xiuRange = [4, 5, 6, 7, 8, 9, 10];
    const taiRange = [11, 12, 13, 14, 15, 16, 17];
    const range = txConstraint === 'T' ? taiRange : xiuRange;

    if (history.length < 20) {
        return {
            top3: [range[0], range[Math.floor(range.length / 2)], range[range.length - 1]],
            top5: range.slice(0, 5),
            probabilities: buildUniformProbs(range),
        };
    }

    const a1 = analysis_frequency(history, txConstraint, range);
    const a2 = analysis_gap(history, txConstraint, range);
    const a3 = analysis_cluster(history, txConstraint, range);
    const a4 = analysis_markov(history, txConstraint, range);
    const a5 = analysis_pair(history, txConstraint, range);

    const weights = { frequency: 0.30, gap: 0.15, cluster: 0.20, markov: 0.20, pair: 0.15 };

    const combined = {};
    range.forEach(s => {
        combined[s] =
            (a1[s] || 0) * weights.frequency +
            (a2[s] || 0) * weights.gap +
            (a3[s] || 0) * weights.cluster +
            (a4[s] || 0) * weights.markov +
            (a5[s] || 0) * weights.pair;
    });

    const total = Object.values(combined).reduce((a, b) => a + b, 0) || 1;
    range.forEach(s => { combined[s] /= total; });

    const ranked = Object.entries(combined)
        .map(([s, p]) => ({ score: parseInt(s), prob: p }))
        .sort((a, b) => b.prob - a.prob);

    const top3 = selectDiverseTop3(ranked, range);
    const top5 = ranked.slice(0, 5).map(x => x.score).sort((a, b) => a - b);

    return {
        top3: top3.sort((a, b) => a - b),
        top5,
        probabilities: combined,
    };
}

function analysis_frequency(history, txConstraint, range) {
    const result = {};
    range.forEach(s => { result[s] = 0; });
    const lastN = history.slice(-100);
    let totalW = 0;
    lastN.forEach((h, i) => {
        if (h.tx === txConstraint && range.includes(h.total)) {
            const age = lastN.length - i - 1;
            result[h.total] += Math.exp(-age / 25);
            totalW += 1;
        }
    });
    range.forEach(s => { result[s] = (result[s] + 0.5) / (totalW + range.length * 0.5); });
    return result;
}

function analysis_gap(history, txConstraint, range) {
    const result = {};
    range.forEach(s => {
        let lastSeen = -1;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].tx === txConstraint && history[i].total === s) { lastSeen = i; break; }
        }
        if (lastSeen === -1) result[s] = 0.9;
        else {
            const gap = history.length - lastSeen;
            if (gap >= 5 && gap <= 15) result[s] = 0.7 + (gap - 5) * 0.02;
            else if (gap < 5) result[s] = 0.3;
            else result[s] = Math.max(0.2, 0.9 - (gap - 15) * 0.03);
        }
    });
    const total = Object.values(result).reduce((a, b) => a + b, 0) || 1;
    range.forEach(s => { result[s] /= total; });
    return result;
}

function analysis_cluster(history, txConstraint, range) {
    const result = {};
    range.forEach(s => { result[s] = 0; });
    const recent = history.filter(h => h.tx === txConstraint).slice(-30).map(h => h.total);
    if (recent.length < 5) { range.forEach(s => { result[s] = 1 / range.length; }); return result; }
    recent.forEach(s => { if (result[s] !== undefined) result[s]++; });
    const boosted = { ...result };
    for (const s of recent) {
        if (boosted[s - 1] !== undefined) boosted[s - 1] += 0.3;
        if (boosted[s + 1] !== undefined) boosted[s + 1] += 0.3;
    }
    const total = Object.values(boosted).reduce((a, b) => a + b, 0) || 1;
    range.forEach(s => { result[s] = boosted[s] / total; });
    return result;
}

function analysis_markov(history, txConstraint, range) {
    const result = {};
    range.forEach(s => { result[s] = 0; });
    const transitions = {};
    for (let i = 2; i < history.length; i++) {
        const h1 = history[i - 2], h2 = history[i - 1], h3 = history[i];
        if (h1.tx !== txConstraint || h2.tx !== txConstraint || h3.tx !== txConstraint) continue;
        if (!range.includes(h1.total) || !range.includes(h2.total)) continue;
        const key = `${h1.total},${h2.total}`;
        transitions[key] = transitions[key] || {};
        transitions[key][h3.total] = (transitions[key][h3.total] || 0) + 1;
    }
    const filtered = history.filter(h => h.tx === txConstraint);
    if (filtered.length < 2) { range.forEach(s => { result[s] = 1 / range.length; }); return result; }
    const last2 = filtered.slice(-2);
    const key = `${last2[0].total},${last2[1].total}`;
    const trans = transitions[key];
    if (trans && Object.keys(trans).length > 0) {
        const total = Object.values(trans).reduce((a, b) => a + b, 0);
        for (const s of range) result[s] = (trans[s] || 0) / total;
    } else {
        range.forEach(s => { result[s] = 1 / range.length; });
    }
    range.forEach(s => { result[s] = result[s] * 0.8 + 0.2 / range.length; });
    return result;
}

function analysis_pair(history, txConstraint, range) {
    const result = {};
    range.forEach(s => { result[s] = 0; });
    const pairs = {};
    for (let i = 1; i < history.length; i++) {
        const h1 = history[i - 1], h2 = history[i];
        if (h1.tx !== txConstraint || h2.tx !== txConstraint) continue;
        if (!range.includes(h1.total) || !range.includes(h2.total)) continue;
        pairs[h1.total] = pairs[h1.total] || {};
        pairs[h1.total][h2.total] = (pairs[h1.total][h2.total] || 0) + 1;
    }
    const filtered = history.filter(h => h.tx === txConstraint);
    if (filtered.length < 1) { range.forEach(s => { result[s] = 1 / range.length; }); return result; }
    const lastTotal = filtered[filtered.length - 1].total;
    const nextCounts = pairs[lastTotal];
    if (nextCounts) {
        const total = Object.values(nextCounts).reduce((a, b) => a + b, 0);
        for (const s of range) result[s] = (nextCounts[s] || 0) / total;
    } else {
        range.forEach(s => { result[s] = 1 / range.length; });
    }
    range.forEach(s => { result[s] = result[s] * 0.7 + 0.3 / range.length; });
    return result;
}

function selectDiverseTop3(ranked, range) {
    const selected = [];
    const mid = Math.floor((range[0] + range[range.length - 1]) / 2);
    selected.push(ranked[0].score);
    const lowHalf = range.filter(s => s < mid);
    for (const r of ranked) {
        if (lowHalf.includes(r.score) && !selected.includes(r.score)) { selected.push(r.score); break; }
    }
    const highHalf = range.filter(s => s > mid);
    for (const r of ranked) {
        if (highHalf.includes(r.score) && !selected.includes(r.score)) { selected.push(r.score); break; }
    }
    for (const r of ranked) {
        if (selected.length >= 3) break;
        if (!selected.includes(r.score)) selected.push(r.score);
    }
    return selected;
}

function buildUniformProbs(range) {
    const p = {};
    range.forEach(s => { p[s] = 1 / range.length; });
    return p;
}

// ============================================================================
// PHẦN 7: PREDICTOR CLASS
// ============================================================================

class SicboPredictor {
    constructor() {
        this.history = [];
        this.lastPrediction = null;
    }

    loadHistory(records) {
        this.history = records;
        this.lastPrediction = this.predict();
        console.log(`[Sicbo] Đã load ${records.length} phiên`);
    }

    push(record) {
        // Đánh giá dự đoán trước + cập nhật AI
        if (this.lastPrediction && record.tx !== 'B') {
            prediction_stats.total++;
            const isCorrect = this.lastPrediction.prediction === record.tx;
            if (isCorrect) prediction_stats.correct++;

            // AI cập nhật trọng số
            const historyPrefix = this.history.slice(-80); // 80 phiên gần nhất để train
            updateAiWeights(record, this.lastPrediction, historyPrefix);

            // Log
            prediction_stats.log.push({
                session: record.session,
                predicted: this.lastPrediction.prediction,
                actual: record.tx,
                correct: isCorrect,
                confidence: this.lastPrediction.confidence,
                reason: this.lastPrediction.reason,
                roadName: this.lastPrediction.roadPattern?.name || 'none',
            });
            if (prediction_stats.log.length > 500) {
                prediction_stats.log = prediction_stats.log.slice(-500);
            }
        }

        this.history.push(record);
        this.lastPrediction = this.predict();
    }

    predict() {
        if (this.history.length < 5) {
            return {
                prediction: 'T',
                confidence: 0.5,
                reason: 'Chưa đủ dữ liệu',
                scorePrediction: [7, 8, 9],
                top5: [6, 7, 8, 9, 10],
                probabilities: {},
                roadPattern: null,
                algoVotes: { T: 0, X: 0, none: 0 },
                algoDetails: {},
                roadVote: null,
                aiPatternVote: null,
                aiTransitionVote: null,
            };
        }

        const roadPattern = detectRoadPattern(this.history);
        const ens = ensemblePredict(this.history, roadPattern);
        const pos = predictPositions(this.history, ens.prediction);

        return {
            prediction: ens.prediction,
            confidence: ens.confidence,
            reason: ens.reason,
            roadPattern,
            algoVotes: ens.algoVotes,
            algoDetails: ens.algoDetails,
            roadVote: ens.roadVote,
            aiPatternVote: ens.aiPatternVote,
            aiTransitionVote: ens.aiTransitionVote,
            scorePrediction: pos.top3,
            top5: pos.top5,
            probabilities: pos.probabilities,
        };
    }

    getAccuracy() {
        if (prediction_stats.total === 0) return 0;
        return prediction_stats.correct / prediction_stats.total * 100;
    }
}

const predictor = new SicboPredictor();

// ============================================================================
// PHẦN 8: FETCH API
// ============================================================================

async function fetch_and_process() {
    try {
        const res = await fetch(api_url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        const new_history = parse_lines(data);
        if (new_history.length === 0) return;

        const last = new_history[new_history.length - 1];

        if (!current_session_id) {
            predictor.loadHistory(new_history);
            txh_history = new_history;
            current_session_id = last.session;
            console.log(`✅ Lần đầu: ${new_history.length} phiên, cuối ${current_session_id}`);
            logPrediction();
        } else if (last.session > current_session_id) {
            const newRecords = new_history.filter(r => r.session > current_session_id);
            for (const rec of newRecords) {
                predictor.push(rec);
                txh_history.push(rec);
            }
            if (txh_history.length > 500) txh_history = txh_history.slice(-500);
            current_session_id = last.session;
            logPrediction();
        }
    } catch (e) {
        console.error("❌ Fetch error:", e.message);
    }
}

function logPrediction() {
    const p = predictor.lastPrediction;
    if (!p) return;
    const next = current_session_id ? current_session_id + 1 : '?';
    const acc = predictor.getAccuracy().toFixed(1);
    const aiAcc = getOverallAccuracy();

    console.log(
        `🔮 ${next}: ${p.prediction} | Vị [${p.scorePrediction.join('-')}] | ` +
        `Top5 [${p.top5.join(',')}] | ${(p.confidence * 100).toFixed(0)}% | ` +
        `Acc: ${acc}% | AI: ${aiAcc}% | ${p.reason}`
    );
}

// ============================================================================
// PHẦN 9: API SERVER
// ============================================================================

const app = fastify({ logger: false });
await app.register(cors, { origin: "*" });

app.get("/api/sicbo/sunwin", async () => {
    const last = txh_history[txh_history.length - 1] || null;
    const pred = predictor.lastPrediction;

    if (!last || !pred) {
        return {
            id: "Sicbo Sunwin AI v5.0",
            status: "waiting",
            phien_hien_tai: current_session_id ? current_session_id + 1 : null,
        };
    }

    // Top thuật toán đang hoạt động tốt
    const topAlgos = Object.entries(aiState.algoWeights)
        .map(([id, w]) => {
            const stats = aiState.algoStats[id] || { total: 0, correct: 0 };
            const acc = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : 'N/A';
            return { id, weight: w.toFixed(3), accuracy: acc + '%', samples: stats.total };
        })
        .sort((a, b) => parseFloat(b.weight) - parseFloat(a.weight))
        .slice(0, 5);

    return {
        id: "Sicbo Sunwin AI v5.0",
        phien_truoc: last.session,
        xuc_xac1: last.dice[0],
        xuc_xac2: last.dice[1],
        xuc_xac3: last.dice[2],
        tong: last.total,
        ket_qua: last.result.toLowerCase(),
        phien_hien_tai: last.session + 1,

        // Dự đoán T/X
        du_doan: pred.prediction === 'T' ? 'tài' : 'xỉu',
        do_tin_cay: (pred.confidence * 100).toFixed(0) + '%',
        ly_do: pred.reason,

        // Dự đoán vị
        du_doan_vi: pred.scorePrediction.join('-'),
        du_doan_top5: pred.top5.join('-'),
        xac_suat_vi: Object.fromEntries(
            Object.entries(pred.probabilities).map(([s, p]) => [s, (p * 100).toFixed(2) + '%'])
        ),

        // Cầu
        cau: pred.roadPattern ? pred.roadPattern.name : 'không rõ',
        cau_type: pred.roadPattern ? pred.roadPattern.type : null,
        cau_strength: pred.roadPattern ? pred.roadPattern.strength.toFixed(2) : null,

        // Votes chi tiết
        algo_votes: pred.algoVotes,
        road_vote: pred.roadVote ? {
            pred: pred.roadVote.pred,
            weight: pred.roadVote.weight.toFixed(2),
        } : null,
        ai_pattern_vote: pred.aiPatternVote ? {
            pred: pred.aiPatternVote.prediction,
            confidence: pred.aiPatternVote.confidence.toFixed(2),
        } : null,
        ai_transition_vote: pred.aiTransitionVote ? {
            pred: pred.aiTransitionVote.prediction,
            source: pred.aiTransitionVote.source,
            confidence: pred.aiTransitionVote.confidence.toFixed(2),
        } : null,

        // AI State
        ai: {
            overall_accuracy: getOverallAccuracy() + '%',
            total_predictions: aiState.total,
            top_algos: topAlgos,
        },

        // Stats
        accuracy: predictor.getAccuracy().toFixed(1) + '%',
    };
});

app.get("/api/sicbo/history", async () => {
    if (!txh_history.length) return { message: "chưa có dữ liệu" };
    return [...txh_history].sort((a, b) => b.session - a.session).map(i => {
        const log = prediction_stats.log.find(l => l.session === i.session);
        return {
            session: i.session,
            dice: i.dice,
            total: i.total,
            result: i.result.toLowerCase(),
            tx_label: i.tx.toLowerCase(),
            du_doan_truoc: log ? (log.predicted === 'T' ? 'tài' : 'xỉu') : null,
            dung: log ? log.correct : null,
            do_tin_cay: log ? (log.confidence * 100).toFixed(0) + '%' : null,
            ly_do: log ? log.reason : null,
            cau: log ? log.roadName : null,
        };
    });
});

app.get("/api/sicbo/stats", async () => {
    const byAlgo = {};
    for (const alg of all_algs) {
        const s = aiState.algoStats[alg.id] || { total: 0, correct: 0 };
        byAlgo[alg.id] = {
            weight: (aiState.algoWeights[alg.id] || 1).toFixed(3),
            total: s.total,
            correct: s.correct,
            accuracy: s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + '%' : 'N/A',
        };
    }

    const byRoad = {};
    for (const [name, s] of Object.entries(aiState.roadStats)) {
        byRoad[name] = {
            weight: (aiState.roadWeights[name] || 1).toFixed(3),
            total: s.total,
            correct: s.correct,
            accuracy: s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + '%' : 'N/A',
        };
    }

    const byConfidence = {};
    for (const [k, v] of Object.entries(aiState.confidenceBuckets)) {
        byConfidence[k] = {
            total: v.total,
            correct: v.correct,
            accuracy: v.total > 0 ? (v.correct / v.total * 100).toFixed(1) + '%' : 'N/A',
        };
    }

    return {
        tong_phien: txh_history.length,
        phien_hien_tai: current_session_id,
        do_chinh_xac: predictor.getAccuracy().toFixed(1) + '%',
        ai_accuracy: getOverallAccuracy() + '%',
        ai_total: aiState.total,
        so_thuat_toan: all_algs.length,
        so_cau: 8,
        thong_ke_thuat_toan: byAlgo,
        thong_ke_cau: byRoad,
        thong_ke_tin_cay: byConfidence,
        chuyen_doi_cau: aiState.roadTransitions,
    };
});

app.get("/api/sicbo/log", async () => {
    return {
        total: prediction_stats.log.length,
        entries: prediction_stats.log.slice(-100),
    };
});

app.get("/api/sicbo/ai", async () => {
    return {
        total: aiState.total,
        correct: aiState.correct,
        accuracy: getOverallAccuracy() + '%',
        algo_weights: aiState.algoWeights,
        road_weights: aiState.roadWeights,
        pattern_count: Object.keys(aiState.patternWeights).length,
        top_patterns: Object.entries(aiState.patternWeights)
            .map(([p, d]) => ({
                pattern: p,
                total: d.total,
                T: d.T,
                X: d.X,
                bias: d.total > 0 ? ((d.T - d.X) / d.total).toFixed(2) : '0',
            }))
            .sort((a, b) => b.total - a.total)
            .slice(0, 10),
    };
});

app.post("/api/sicbo/ai/reset", async () => {
    aiState.algoWeights = {};
    aiState.roadWeights = {};
    aiState.algoStats = {};
    aiState.roadStats = {};
    aiState.roadTransitions = {};
    aiState.patternWeights = {};
    aiState.total = 0;
    aiState.correct = 0;
    for (const k of Object.keys(aiState.confidenceBuckets)) {
        aiState.confidenceBuckets[k] = { total: 0, correct: 0 };
    }
    initAiWeights();
    saveAiState();
    return { status: "reset_ok" };
});

app.get("/", async () => {
    return {
        status: "ok",
        version: "5.0",
        msg: "Sicbo Sunwin AI v5.0 - 10 Algo + 8 Road + AI Self-Learning",
        endpoints: {
            "/api/sicbo/sunwin": "Dự đoán + Cầu + Vị + AI",
            "/api/sicbo/history": "Lịch sử + đánh giá",
            "/api/sicbo/stats": "Thống kê chi tiết",
            "/api/sicbo/log": "Log dự đoán",
            "/api/sicbo/ai": "Trạng thái AI (weights, patterns)",
            "POST /api/sicbo/ai/reset": "Reset AI",
        },
        thong_tin: {
            so_thuat_toan: 10,
            so_cau: 8,
            ai_accuracy: getOverallAccuracy() + '%',
            tong_du_doan_ai: aiState.total,
            phien_hien_tai: current_session_id,
        },
    };
});

// ============================================================================
// PHẦN 10: KHỞI ĐỘNG
// ============================================================================

const start = async () => {
    // Load AI state trước
    loadAiState();
    initAiWeights();

    try {
        await app.listen({ port, host: "0.0.0.0" });
    } catch (err) {
        const errMsg = `
=========== SERVER ERROR ===========
time: ${new Date().toISOString()}
error: ${err.message}
stack: ${err.stack}
====================================
`;
        console.error(errMsg);
        fs.writeFileSync(path.join(__dirname, "server-error.log"), errMsg, { flag: "a+" });
        process.exit(1);
    }

    console.log("\n" + "=".repeat(60));
    console.log("🚀 SICBO SUNWIN AI SERVER v5.0");
    console.log("=".repeat(60));
    console.log(`   ➜ Local:   http://localhost:${port}/`);
    console.log(`   ➜ AI Accuracy: ${getOverallAccuracy()}% (${aiState.total} mẫu)`);
    console.log("\n📌 API ENDPOINTS:");
    console.log(`   ➜ GET  /api/sicbo/sunwin   → Dự đoán + Cầu + Vị + AI`);
    console.log(`   ➜ GET  /api/sicbo/history  → Lịch sử + đánh giá`);
    console.log(`   ➜ GET  /api/sicbo/stats    → Thống kê toàn bộ`);
    console.log(`   ➜ GET  /api/sicbo/log      → Log dự đoán`);
    console.log(`   ➜ GET  /api/sicbo/ai       → Trạng thái AI`);
    console.log(`   ➜ POST /api/sicbo/ai/reset → Reset AI`);
    console.log("\n🎯 KIẾN TRÚC:");
    console.log(`   • 10 thuật toán cũ (frequency, markov, ngram, neo_pattern,`); 
    console.log(`     deep_analysis, transformer, super_bridge, adaptive_markov,`);
    console.log(`     neural_pattern, quantum)`);
    console.log(`   • 8 loại cầu (1-1, 2-2, 3-2, 3-1, 2-1, Đôi, Thang Lời, Nghiêng)`);
    console.log(`   • AI tự học: cập nhật weight từng thuật toán theo độ chính xác`);
    console.log(`   • AI tự học: cập nhật weight từng loại cầu`);
    console.log(`   • AI tự học: ma trận chuyển đổi cầu (cầu A → cầu B)`);
    console.log(`   • AI tự học: pattern 5 phiên gần nhất`);
    console.log(`   • Dự đoán vị: 5 phân tích kết hợp (freq, gap, cluster, markov, pair)`);
    console.log("=".repeat(60));
};

fetch_and_process();
clearInterval(fetch_interval);
fetch_interval = setInterval(fetch_and_process, 5000);
start();
