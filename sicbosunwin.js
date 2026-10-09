// ============================================================================
// SICBO SUNWIN AI SERVER v4.0 - BẢN HOÀN CHỈNH
// Tích hợp: Cầu + Ensemble 7 Model + Bayes + Dự đoán vị chi tiết
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

// ============================================================================
// GLOBAL STATE
// ============================================================================
let txh_history = [];
let current_session_id = null;
let fetch_interval = null;
let last_prediction = null;

const prediction_stats = {
    total: 0,
    correct: 0,
    log: [],
    byRoad: {},      // Thống kê theo loại cầu
    byConfidence: {  // Thống kê theo độ tin cậy
        high: { total: 0, correct: 0 },   // > 70%
        medium: { total: 0, correct: 0 }, // 60-70%
        low: { total: 0, correct: 0 },    // < 60%
    },
};

// ============================================================================
// PHẦN 1: PARSE DỮ LIỆU TỪ API
// ============================================================================
function parse_lines(data) {
    if (!data || !data.data || !Array.isArray(data.data.resultList)) return [];

    const sorted = data.data.resultList.sort((a, b) => {
        const id_a = parseInt(a.gameNum.slice(1));
        const id_b = parseInt(b.gameNum.slice(1));
        return id_b - id_a;
    });

    return sorted.map(item => {
        const total = item.score;
        let tx, result_text;

        if (total >= 4 && total <= 10) {
            tx = 'X';
            result_text = "XIU";
        } else if (total >= 11 && total <= 17) {
            tx = 'T';
            result_text = "TAI";
        } else if (total === 3 || total === 18) {
            tx = 'B';
            result_text = "BAO";
        } else {
            tx = 'N';
            result_text = "UNKNOWN";
        }

        const dice = Array.isArray(item.facesList)
            ? item.facesList
            : (typeof item.keyR === 'string' ? item.keyR.split('-').map(Number) : [0, 0, 0]);

        return {
            session: parseInt(item.gameNum.slice(1)),
            dice,
            total,
            result: result_text,
            tx,
        };
    }).sort((a, b) => a.session - b.session);
}

// ============================================================================
// PHẦN 2: NHẬN DIỆN CẦU (8 LOẠI)
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
            name: '1-1',
            type: 'theo',
            prediction: last === 'T' ? 'X' : 'T',
            strength: Math.min(0.85, 0.4 + count * 0.08),
            desc: `Theo cầu 1-1 (${count} phiên xen kẽ)`,
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
            return {
                name: '2-2',
                type: 'theo',
                prediction: last.val,
                strength: 0.72,
                desc: `Theo cầu 2-2 (${last.len}/2)`,
            };
        } else {
            return {
                name: '2-2',
                type: 'be',
                prediction: last.val === 'T' ? 'X' : 'T',
                strength: 0.78,
                desc: `Bẻ cầu 2-2 (đủ block)`,
            };
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
            return {
                name: `${lenA}-${lenB}`,
                type: 'theo',
                prediction: last.val,
                strength: 0.75,
                desc: `Theo cầu ${lenA}-${lenB} (${last.len}/${expected})`,
            };
        } else {
            return {
                name: `${lenA}-${lenB}`,
                type: 'be',
                prediction: last.val === 'T' ? 'X' : 'T',
                strength: 0.80,
                desc: `Bẻ cầu ${lenA}-${lenB} → đảo`,
            };
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
        return {
            name: `Đôi ${last}${last}`,
            type: 'be',
            prediction: last === 'T' ? 'X' : 'T',
            strength: 0.68,
            desc: `Bẻ cầu Đôi ${last}${last} → dự ${last === 'T' ? 'X' : 'T'}`,
        };
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
            desc: shouldBreak
                ? `Bẻ cầu Thang Lời ${count} phiên → đảo`
                : `Theo cầu Thang Lời ${count} phiên`,
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
        return {
            name: 'Nghiêng',
            type: 'be',
            prediction: t > x ? 'X' : 'T',
            strength: 0.62,
            desc: `Bẻ cầu Nghiêng ${t > x ? 'Tài' : 'Xỉu'} ${Math.max(t, x)}/${recent.length}`,
        };
    }
    return null;
}

function buildBlocks(seq) {
    const blocks = [];
    let cur = { val: seq[seq.length - 1], len: 1 };
    for (let i = seq.length - 2; i >= 0; i--) {
        if (seq[i] === cur.val) cur.len++;
        else {
            blocks.unshift(cur);
            cur = { val: seq[i], len: 1 };
        }
    }
    blocks.unshift(cur);
    return blocks;
}

// ============================================================================
// PHẦN 3: 7 MODEL ENSEMBLE
// ============================================================================

function model_frequency(seq) {
    if (seq.length < 15) return null;
    const r10 = seq.slice(-10);
    const t = r10.filter(x => x === 'T').length;
    const x = r10.filter(x => x === 'X').length;
    if (t >= 7) return 'X';
    if (x >= 7) return 'T';
    return null;
}

function model_markov(seq) {
    if (seq.length < 20) return null;
    let bestPred = null, bestConf = 0;

    for (let order = 2; order <= 4; order++) {
        if (seq.length < order + 5) continue;
        const trans = {};
        for (let i = 0; i <= seq.length - order - 1; i++) {
            const key = seq.slice(i, i + order).join('');
            const next = seq[i + order];
            trans[key] = trans[key] || { T: 0, X: 0 };
            trans[key][next]++;
        }
        const lastKey = seq.slice(-order).join('');
        const counts = trans[lastKey];
        if (counts && counts.T + counts.X >= 3) {
            const tot = counts.T + counts.X;
            const conf = Math.abs(counts.T - counts.X) / tot;
            if (conf > bestConf && conf > 0.55) {
                bestConf = conf;
                bestPred = counts.T > counts.X ? 'T' : 'X';
            }
        }
    }
    return bestPred;
}

function model_ngram(seq) {
    for (let k = 3; k <= 5; k++) {
        if (seq.length < k + 5) continue;
        const last = seq.slice(-k).join('');
        let t = 0, x = 0;
        for (let i = 0; i <= seq.length - k - 1; i++) {
            if (seq.slice(i, i + k).join('') === last) {
                if (seq[i + k] === 'T') t++;
                else x++;
            }
        }
        const tot = t + x;
        if (tot >= 3 && Math.abs(t - x) / tot >= 0.6) {
            return t > x ? 'T' : 'X';
        }
    }
    return null;
}

function model_run(seq) {
    if (seq.length < 5) return null;
    const runs = [];
    let cur = seq[0], len = 1;
    for (let i = 1; i < seq.length; i++) {
        if (seq[i] === cur) len++;
        else {
            runs.push({ val: cur, len });
            cur = seq[i];
            len = 1;
        }
    }
    runs.push({ val: cur, len });

    const last = runs[runs.length - 1];
    const prev = runs[runs.length - 2];

    if (last.len >= 4) return last.val === 'T' ? 'X' : 'T';
    if (last.len === 1 && prev && prev.len === 1) return last.val === 'T' ? 'X' : 'T';
    return null;
}

function model_distribution(history) {
    const f = history.filter(h => h.tx !== 'B');
    if (f.length < 30) return null;
    const recent = f.slice(-30);
    const avg = recent.reduce((a, b) => a + b.total, 0) / recent.length;
    if (avg > 12.5) return 'X';
    if (avg < 8.5) return 'T';
    return null;
}

function model_bayes(seq) {
    if (seq.length < 10) return null;
    const recent = seq.slice(-10);
    const t = recent.filter(x => x === 'T').length;
    const x = recent.filter(x => x === 'X').length;
    const pT = (t + 1) / (recent.length + 2);
    const pX = (x + 1) / (recent.length + 2);
    const postT = pT * 0.5;
    const postX = pX * 0.5;
    if (Math.abs(postT - postX) > 0.15) return postT > postX ? 'T' : 'X';
    return null;
}

function model_alternating(seq) {
    if (seq.length < 6) return null;
    let count = 0;
    for (let i = seq.length - 1; i > 0; i--) {
        if (seq[i] !== seq[i - 1]) count++;
        else break;
    }
    if (count >= 3) return seq[seq.length - 1] === 'T' ? 'X' : 'T';
    return null;
}

// ============================================================================
// PHẦN 4: ENSEMBLE VOTING
// ============================================================================

function ensemblePredict(history, roadPattern) {
    const seq = history.filter(h => h.tx !== 'B').map(h => h.tx);

    const models = [
        { name: 'frequency', pred: model_frequency(seq), weight: 1.0 },
        { name: 'markov', pred: model_markov(seq), weight: 1.2 },
        { name: 'ngram', pred: model_ngram(seq), weight: 1.1 },
        { name: 'run', pred: model_run(seq), weight: 1.0 },
        { name: 'distribution', pred: model_distribution(history), weight: 0.9 },
        { name: 'bayes', pred: model_bayes(seq), weight: 1.3 },
        { name: 'alternating', pred: model_alternating(seq), weight: 1.0 },
    ];

    let tScore = 0, xScore = 0;
    const votes = { T: 0, X: 0, none: 0 };

    for (const m of models) {
        if (m.pred === 'T') { tScore += m.weight; votes.T++; }
        else if (m.pred === 'X') { xScore += m.weight; votes.X++; }
        else votes.none++;
    }

    // Ưu tiên cầu
    if (roadPattern) {
        const roadWeight = 2.0;
        if (roadPattern.prediction === 'T') tScore += roadWeight * roadPattern.strength;
        else xScore += roadWeight * roadPattern.strength;
    }

    const total = tScore + xScore;
    if (total === 0) {
        return {
            prediction: 'T',
            confidence: 0.5,
            votes,
            reason: 'Fallback cân bằng',
        };
    }

    const prediction = tScore > xScore ? 'T' : 'X';
    const confidence = Math.max(tScore, xScore) / total;

    let reason;
    if (roadPattern) {
        reason = roadPattern.desc;
    } else if (votes.T === 0 && votes.X > 0) {
        reason = `Ensemble (X thắng tuyệt đối)`;
    } else if (votes.X === 0 && votes.T > 0) {
        reason = `Ensemble (T thắng tuyệt đối)`;
    } else if (confidence > 0.65) {
        reason = `Ensemble | Bayes ${confidence.toFixed(2)}`;
    } else {
        reason = `Ensemble | Fallback theo nhịp`;
    }

    return { prediction, confidence, votes, reason };
}

// ============================================================================
// PHẦN 5: DỰ ĐOÁN VỊ (SCORE PREDICTOR)
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
            method: 'insufficient_data',
        };
    }

    const a1 = analysis_frequency(history, txConstraint, range);
    const a2 = analysis_gap(history, txConstraint, range);
    const a3 = analysis_cluster(history, txConstraint, range);
    const a4 = analysis_markov(history, txConstraint, range);
    const a5 = analysis_pair(history, txConstraint, range);

    const weights = {
        frequency: 0.30,
        gap: 0.15,
        cluster: 0.20,
        markov: 0.20,
        pair: 0.15,
    };

    const combined = {};
    range.forEach(s => { combined[s] = 0; });

    for (const s of range) {
        combined[s] =
            (a1[s] || 0) * weights.frequency +
            (a2[s] || 0) * weights.gap +
            (a3[s] || 0) * weights.cluster +
            (a4[s] || 0) * weights.markov +
            (a5[s] || 0) * weights.pair;
    }

    const totalWeight = Object.values(combined).reduce((a, b) => a + b, 0) || 1;
    range.forEach(s => { combined[s] /= totalWeight; });

    const ranked = Object.entries(combined)
        .map(([s, p]) => ({ score: parseInt(s), prob: p }))
        .sort((a, b) => b.prob - a.prob);

    const top3 = selectDiverseTop3(ranked, range);
    const top5 = ranked.slice(0, 5).map(x => x.score).sort((a, b) => a - b);

    return {
        top3: top3.sort((a, b) => a - b),
        top5,
        probabilities: combined,
        ranked,
        method: 'weighted_ensemble',
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
            const w = Math.exp(-age / 25);
            result[h.total] += w;
            totalW += w;
        }
    });

    range.forEach(s => {
        result[s] = (result[s] + 0.5) / (totalW + range.length * 0.5);
    });

    return result;
}

function analysis_gap(history, txConstraint, range) {
    const result = {};
    range.forEach(s => {
        let lastSeen = -1;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].tx === txConstraint && history[i].total === s) {
                lastSeen = i;
                break;
            }
        }

        if (lastSeen === -1) {
            result[s] = 0.9;
        } else {
            const gap = history.length - lastSeen;
            if (gap >= 5 && gap <= 15) {
                result[s] = 0.7 + (gap - 5) * 0.02;
            } else if (gap < 5) {
                result[s] = 0.3;
            } else {
                result[s] = Math.max(0.2, 0.9 - (gap - 15) * 0.03);
            }
        }
    });

    const total = Object.values(result).reduce((a, b) => a + b, 0) || 1;
    range.forEach(s => { result[s] /= total; });
    return result;
}

function analysis_cluster(history, txConstraint, range) {
    const result = {};
    range.forEach(s => { result[s] = 0; });

    const recent = history
        .filter(h => h.tx === txConstraint)
        .slice(-30)
        .map(h => h.total);

    if (recent.length < 5) {
        range.forEach(s => { result[s] = 1 / range.length; });
        return result;
    }

    recent.forEach(s => {
        if (result[s] !== undefined) result[s]++;
    });

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
    if (filtered.length < 2) {
        range.forEach(s => { result[s] = 1 / range.length; });
        return result;
    }

    const last2 = filtered.slice(-2);
    const key = `${last2[0].total},${last2[1].total}`;
    const trans = transitions[key];

    if (trans && Object.keys(trans).length > 0) {
        const total = Object.values(trans).reduce((a, b) => a + b, 0);
        for (const s of range) {
            result[s] = (trans[s] || 0) / total;
        }
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
    if (filtered.length < 1) {
        range.forEach(s => { result[s] = 1 / range.length; });
        return result;
    }

    const lastTotal = filtered[filtered.length - 1].total;
    const nextCounts = pairs[lastTotal];

    if (nextCounts) {
        const total = Object.values(nextCounts).reduce((a, b) => a + b, 0);
        for (const s of range) {
            result[s] = (nextCounts[s] || 0) / total;
        }
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
        if (lowHalf.includes(r.score) && !selected.includes(r.score)) {
            selected.push(r.score);
            break;
        }
    }

    const highHalf = range.filter(s => s > mid);
    for (const r of ranked) {
        if (highHalf.includes(r.score) && !selected.includes(r.score)) {
            selected.push(r.score);
            break;
        }
    }

    for (const r of ranked) {
        if (selected.length >= 3) break;
        if (!selected.includes(r.score)) selected.push(r.score);
    }

    return selected;
}

function buildUniformProbs(range) {
    const probs = {};
    range.forEach(s => { probs[s] = 1 / range.length; });
    return probs;
}

// ============================================================================
// PHẦN 6: PREDICTOR CLASS
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
        // Đánh giá dự đoán trước
        if (this.lastPrediction && record.tx !== 'B') {
            prediction_stats.total++;
            const isCorrect = this.lastPrediction.prediction === record.tx;
            if (isCorrect) prediction_stats.correct++;

            // Thống kê theo độ tin cậy
            const conf = this.lastPrediction.confidence;
            const bucket = conf > 0.70 ? 'high' : conf > 0.60 ? 'medium' : 'low';
            prediction_stats.byConfidence[bucket].total++;
            if (isCorrect) prediction_stats.byConfidence[bucket].correct++;

            // Thống kê theo loại cầu
            const roadName = this.lastPrediction.roadPattern?.name || 'none';
            prediction_stats.byRoad[roadName] = prediction_stats.byRoad[roadName] || { total: 0, correct: 0 };
            prediction_stats.byRoad[roadName].total++;
            if (isCorrect) prediction_stats.byRoad[roadName].correct++;

            // Log
            prediction_stats.log.push({
                session: record.session,
                predicted: this.lastPrediction.prediction,
                actual: record.tx,
                correct: isCorrect,
                confidence: conf,
                reason: this.lastPrediction.reason,
                roadName,
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
                votes: { T: 0, X: 0, none: 0 },
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
            votes: ens.votes,
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
// PHẦN 7: FETCH API
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
            if (txh_history.length > 500) {
                txh_history = txh_history.slice(-500);
            }
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
    console.log(
        `🔮 Phiên ${next}: ${p.prediction} | Vị [${p.scorePrediction.join('-')}] | ` +
        `Top5 [${p.top5.join(',')}] | ${(p.confidence * 100).toFixed(0)}% | ` +
        `Acc: ${acc}% | ${p.reason}`
    );
}

// ============================================================================
// PHẦN 8: API SERVER
// ============================================================================

const app = fastify({ logger: false });
await app.register(cors, { origin: "*" });

app.get("/api/sicbo/sunwin", async () => {
    const last = txh_history[txh_history.length - 1] || null;
    const pred = predictor.lastPrediction;

    if (!last || !pred) {
        return {
            id: "Sicbo Sunwin AI v4.0",
            status: "waiting",
            phien_hien_tai: current_session_id ? current_session_id + 1 : null,
        };
    }

    return {
        id: "Sicbo Sunwin AI v4.0",
        phien_truoc: last.session,
        xuc_xac1: last.dice[0],
        xuc_xac2: last.dice[1],
        xuc_xac3: last.dice[2],
        tong: last.total,
        ket_qua: last.result.toLowerCase(),
        phien_hien_tai: last.session + 1,
        du_doan: pred.prediction === 'T' ? 'tài' : 'xỉu',
        du_doan_vi: pred.scorePrediction.join('-'),
        du_doan_top5: pred.top5.join('-'),
        xac_suat_vi: Object.fromEntries(
            Object.entries(pred.probabilities).map(([s, p]) => [s, (p * 100).toFixed(2) + '%'])
        ),
        do_tin_cay: (pred.confidence * 100).toFixed(0) + '%',
        ly_do: pred.reason,
        cau: pred.roadPattern ? pred.roadPattern.name : 'không rõ',
        votes: pred.votes,
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
        };
    });
});

app.get("/api/sicbo/stats", async () => {
    const acc = prediction_stats.total > 0
        ? (prediction_stats.correct / prediction_stats.total * 100).toFixed(1)
        : "0.0";

    const byConfidence = {};
    for (const [k, v] of Object.entries(prediction_stats.byConfidence)) {
        byConfidence[k] = {
            total: v.total,
            correct: v.correct,
            accuracy: v.total > 0 ? (v.correct / v.total * 100).toFixed(1) + '%' : 'N/A',
        };
    }

    const byRoad = {};
    for (const [k, v] of Object.entries(prediction_stats.byRoad)) {
        byRoad[k] = {
            total: v.total,
            correct: v.correct,
            accuracy: v.total > 0 ? (v.correct / v.total * 100).toFixed(1) + '%' : 'N/A',
        };
    }

    return {
        tong_phien: txh_history.length,
        phien_hien_tai: current_session_id,
        do_chinh_xac: acc + '%',
        tong_du_doan: prediction_stats.total,
        du_doan_dung: prediction_stats.correct,
        so_model: 7,
        so_cau: 8,
        pham_vi_vi: {
            xiu: [4, 5, 6, 7, 8, 9, 10],
            tai: [11, 12, 13, 14, 15, 16, 17],
        },
        thong_ke_theo_tin_cay: byConfidence,
        thong_ke_theo_cau: byRoad,
    };
});

app.get("/api/sicbo/log", async () => {
    return {
        total: prediction_stats.log.length,
        entries: prediction_stats.log.slice(-100),
    };
});

app.get("/", async () => {
    return {
        status: "ok",
        version: "4.0",
        msg: "Sicbo Sunwin AI Server 🚀",
        endpoints: {
            "/api/sicbo/sunwin": "Dự đoán Tài/Xỉu + 3 vị + Top 5",
            "/api/sicbo/history": "Lịch sử + đánh giá",
            "/api/sicbo/stats": "Thống kê chi tiết",
            "/api/sicbo/log": "Log 100 dự đoán",
        },
        thong_tin: {
            so_model: 7,
            so_cau: 8,
            phien_hien_tai: current_session_id,
            tong_phien: txh_history.length,
        },
    };
});

// ============================================================================
// PHẦN 9: KHỞI ĐỘNG
// ============================================================================

const start = async () => {
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
    console.log("🚀 SICBO SUNWIN AI SERVER v4.0");
    console.log("=".repeat(60));
    console.log(`   ➜ Local:   http://localhost:${port}/`);
    console.log("\n📌 API ENDPOINTS:");
    console.log(`   ➜ GET /api/sicbo/sunwin   → Dự đoán + Cầu + Vị + Top5`);
    console.log(`   ➜ GET /api/sicbo/history  → Lịch sử + đánh giá`);
    console.log(`   ➜ GET /api/sicbo/stats    → Thống kê theo cầu + tin cậy`);
    console.log(`   ➜ GET /api/sicbo/log      → Log dự đoán`);
    console.log("\n🎯 THUẬT TOÁN:");
    console.log(`   • 8 loại cầu (1-1, 2-2, 3-2, 3-1, 2-1, Đôi, Thang Lời, Nghiêng)`);
    console.log(`   • 7 model ensemble (frequency, markov, ngram, run, distribution, bayes, alternating)`);
    console.log(`   • Cầu có weight ×2 khi vote`);
    console.log(`   • Dự đoán vị: 5 phân tích (freq, gap, cluster, markov, pair)`);
    console.log("=".repeat(60));
};

fetch_and_process();
clearInterval(fetch_interval);
fetch_interval = setInterval(fetch_and_process, 5000);
start();
