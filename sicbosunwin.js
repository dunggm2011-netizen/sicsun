import fastify from "fastify";
import cors from "@fastify/cors";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import fetch from "node-fetch";
import fs from "fs";

// --- CẤU HÌNH ---
const port = 3000;
const api_url = "https://api.wsktnus8.net/v2/history/getLastResult?gameId=ktrng_3979&size=100&tableId=39791215743193&curPage=1"; 

// --- GLOBAL STATE ---
let txh_history = []; 
let current_session_id = null; 
let fetch_interval = null;
let prediction_memory = { T: [], X: [] }; // Thay thế localStorage

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- UTILITIES ---
function parse_lines(data) {
    if (!data || !data.data || !Array.isArray(data.data.resultList)) return [];
    
    const sorted_list = data.data.resultList.sort((a, b) => {
        const id_a = parseInt(a.gameNum.slice(1));
        const id_b = parseInt(b.gameNum.slice(1));
        return id_b - id_a;
    });

    const arr = sorted_list.map(item => {
        const total = item.score;
        let tx;
        let result_truyen_thong;
        
        if (total >= 4 && total <= 10) {
            tx = 'X';
            result_truyen_thong = "XIU";
        } else if (total >= 11 && total <= 17) {
            tx = 'T';
            result_truyen_thong = "TAI";
        } else if (total === 3 || total === 18) {
            tx = 'B';
            result_truyen_thong = "BAO";
        } else {
            tx = 'N'; 
            result_truyen_thong = "UNKNOWN";
        }
        
        const dice_faces = Array.isArray(item.facesList) ? item.facesList : 
                           (typeof item.keyR === 'string' ? item.keyR.split('-').map(Number) : [0, 0, 0]);

        return {
            session: parseInt(item.gameNum.slice(1)), 
            dice: dice_faces,
            total: total,
            result: result_truyen_thong, 
            tx: tx 
        };
    });

    return arr.sort((a, b) => a.session - b.session);
}

function last_n(arr, n) {
    return arr.slice(Math.max(0, arr.length - n));
}

function majority(obj) {
    let max_k = null,
        max_v = -Infinity;
    for (let k in obj) {
        if (obj[k] > max_v) {
            max_v = obj[k];
            max_k = k;
        }
    }
    return {
        key: max_k,
        val: max_v
    };
}

function sum(nums) {
    return nums.reduce((a, b) => a + b, 0);
}

function avg(nums) {
    return nums.length ? sum(nums) / nums.length : 0;
}

function entropy(arr) {
    if (!arr.length) return 0;
    const freq = arr.reduce((a, v) => {
        a[v] = (a[v] || 0) + 1;
        return a;
    }, {});
    const n = arr.length;
    let e = 0;
    for (let k in freq) {
        const p = freq[k] / n;
        e -= p * Math.log2(p);
    }
    return e;
}

function similarity(a, b) {
    if (a.length !== b.length) return 0;
    let m = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] === b[i]) m++;
    }
    return m / a.length;
}

function extract_features(history) {
    const tx_filtered = history.filter(h => h.tx !== 'B'); 
    const tx = tx_filtered.map(h => h.tx);
    const totals = tx_filtered.map(h => h.total);
    const features = {
        tx,
        totals,
        freq: tx.reduce((a, v) => {
            a[v] = (a[v] || 0) + 1;
            return a;
        }, {})
    };

    let runs = [],
        cur = tx[0],
        len = 1;
    for (let i = 1; i < tx.length; i++) {
        if (tx[i] === cur) len++;
        else {
            runs.push({
                val: cur,
                len
            });
            cur = tx[i];
            len = 1;
        }
    }
    if (tx.length) runs.push({
        val: cur,
        len
    });
    features.runs = runs;
    features.max_run = runs.reduce((m, r) => Math.max(m, r.len), 0) || 0;

    features.mean_total = avg(totals);
    features.std_total = Math.sqrt(avg(totals.map(t => Math.pow(t - features.mean_total, 2))));
    features.entropy = entropy(tx);

    return features;
}

// ==================== THUẬT TOÁN DỰ ĐOÁN TÀI/XỈU - ULTIMATE AI ====================

// 1. THUẬT TOÁN FREQUENCY BALANCE PLUS
function algo5_freq_rebalance(history) {
    if (history.length < 15) return null;
    
    const features = extract_features(history);
    const tx = features.tx;
    
    // Phân tích tổng thể
    const total_t = (features.freq['T'] || 0);
    const total_x = (features.freq['X'] || 0);
    const total_games = tx.length;
    
    if (total_games === 0) return null;
    
    // Phân tích 30 phiên gần nhất
    const recent_30 = tx.slice(-30);
    const recent_t = recent_30.filter(x => x === 'T').length;
    const recent_x = recent_30.filter(x => x === 'X').length;
    
    // Phân tích 10 phiên gần nhất
    const recent_10 = tx.slice(-10);
    const recent_10_t = recent_10.filter(x => x === 'T').length;
    const recent_10_x = recent_10.filter(x => x === 'X').length;
    
    // Quy tắc 1: Xu hướng gần đây mạnh
    if (recent_10_t >= 7) return 'X';
    if (recent_10_x >= 7) return 'T';
    
    // Quy tắc 2: Mất cân bằng dài hạn
    if (total_t > total_x + 8 && recent_t > recent_x + 2) return 'X';
    if (total_x > total_t + 8 && recent_x > recent_t + 2) return 'T';
    
    // Quy tắc 3: Mean reversion
    if (recent_10_t > recent_10_x + 3) return 'X';
    if (recent_10_x > recent_10_t + 3) return 'T';
    
    return null;
}

// 2. THUẬT TOÁN MARKOV ENHANCED
function algoa_markov(history) {
    const tx = extract_features(history).tx;
    if (tx.length < 20) return null;
    
    let best_pred = null;
    let best_confidence = 0;
    
    // Thử nhiều bậc Markov
    for (let order = 2; order <= 5; order++) {
        if (tx.length < order + 5) continue;
        
        const transitions = {};
        
        // Xây dựng ma trận chuyển đổi
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

// 3. THUẬT TOÁN N-GRAM ADVANCED
function algob_ngram(history) {
    const tx = extract_features(history).tx;
    
    for (let k = 3; k <= 6; k++) {
        if (tx.length < k + 10) continue;
        
        const last_gram = tx.slice(-k).join('');
        let counts = { t: 0, x: 0 };
        let total_matches = 0;
        
        // Tìm pattern tương tự
        for (let i = 0; i <= tx.length - k - 1; i++) {
            const gram = tx.slice(i, i + k).join('');
            if (gram === last_gram) {
                const next_val = tx[i + k].toLowerCase();
                counts[next_val]++;
                total_matches++;
            }
        }
        
        // Cần ít nhất 3 matches để có độ tin cậy
        if (total_matches >= 3) {
            const ratio = Math.abs(counts.t - counts.x) / total_matches;
            if (ratio >= 0.6) {
                return counts.t > counts.x ? 'T' : 'X';
            }
        }
    }
    
    return null;
}

// 4. THUẬT TOÁN NEO PATTERN ULTIMATE
function algos_neo_pattern(history) {
    const tx = extract_features(history).tx;
    const len = tx.length;
    
    if (len < 40) return null;
    
    const pattern_lengths = [4, 5, 6];
    let best_pred = null;
    let max_confidence = 0;
    
    for (let pat_len of pattern_lengths) {
        if (len < pat_len * 2) continue;
        
        const target_pattern = tx.slice(-pat_len);
        let pattern_counts = { t: 0, x: 0 };
        
        for (let i = 0; i <= len - pat_len - 1; i++) {
            const hist_pattern = tx.slice(i, i + pat_len);
            
            // Tính độ tương đồng pattern
            let match_count = 0;
            for (let j = 0; j < pat_len; j++) {
                if (hist_pattern[j] === target_pattern[j]) match_count++;
            }
            const similarity = match_count / pat_len;
            
            if (similarity >= 0.8) {
                const next_result = tx[i + pat_len];
                pattern_counts[next_result.toLowerCase()]++;
            }
        }
        
        const total_matches = pattern_counts.t + pattern_counts.x;
        if (total_matches >= 5) {
            const t_ratio = pattern_counts.t / total_matches;
            const x_ratio = pattern_counts.x / total_matches;
            const confidence = Math.abs(t_ratio - x_ratio);
            
            if (confidence > max_confidence && confidence > 0.6) {
                max_confidence = confidence;
                best_pred = pattern_counts.t > pattern_counts.x ? 'T' : 'X';
            }
        }
    }
    
    return best_pred;
}

// 5. THUẬT TOÁN DEEP LEARNING AI
function algof_super_deep_analysis(history) {
    if (history.length < 80) return null;
    
    const features = extract_features(history);
    const tx = features.tx;
    const totals = features.totals;
    const runs = features.runs;
    
    // Layer 1: Trend Analysis
    const recent_50 = tx.slice(-50);
    const recent_50_t = recent_50.filter(x => x === 'T').length;
    const recent_50_x = recent_50.filter(x => x === 'X').length;
    
    // Layer 2: Total Analysis
    const recent_totals = totals.slice(-30);
    const avg_recent = avg(recent_totals);
    
    // Layer 3: Pattern Recognition
    let pattern_pred = null;
    const last_10 = tx.slice(-10);
    const patterns = {
        'TTXTT': 'X',
        'TXTXT': 'X',
        'XXTXX': 'T',
        'XTXTX': 'T',
        'TTTTX': 'X',
        'XXXXT': 'T'
    };
    
    const last_5_pattern = last_10.slice(-5).join('');
    if (patterns[last_5_pattern]) {
        pattern_pred = patterns[last_5_pattern];
    }
    
    // Layer 4: Run Analysis
    let run_pred = null;
    if (runs.length >= 2) {
        const last_run = runs[runs.length - 1];
        if (last_run.len >= 4) {
            run_pred = last_run.val === 'T' ? 'X' : 'T';
        } else if (last_run.len === 1) {
            const last_3_runs = runs.slice(-3);
            if (last_3_runs.length === 3 && last_3_runs.every(r => r.len === 1)) {
                run_pred = last_run.val === 'T' ? 'X' : 'T';
            }
        }
    }
    
    // Tổng hợp tất cả layers
    const votes = [];
    
    if (Math.abs(recent_50_t - recent_50_x) > 10) {
        votes.push(recent_50_t > recent_50_x ? 'T' : 'X');
    }
    
    if (avg_recent > 13.0) {
        votes.push('X');
    } else if (avg_recent < 8.5) {
        votes.push('T');
    }
    
    if (pattern_pred) votes.push(pattern_pred);
    if (run_pred) votes.push(run_pred);
    
    if (votes.length === 0) return null;
    
    // Đếm votes
    const t_votes = votes.filter(x => x === 'T').length;
    const x_votes = votes.filter(x => x === 'X').length;
    
    if (t_votes > x_votes && t_votes >= 2) return 'T';
    if (x_votes > t_votes && x_votes >= 2) return 'X';
    
    return null;
}

// 6. THUẬT TOÁN TRANSFORMER PRO
function algoe_transformer(history) {
    const tx = extract_features(history).tx;
    const len = tx.length;
    
    if (len < 60) return null;
    
    const window_size = 10;
    const target_seq = tx.slice(-window_size).join('');
    
    let t_weight = 0, x_weight = 0;
    let total_similarity = 0;
    
    for (let i = 0; i <= len - window_size - 1; i++) {
        const hist_seq = tx.slice(i, i + window_size).join('');
        const similarity_score = similarity(hist_seq, target_seq);
        
        if (similarity_score > 0.7) {
            const next_result = tx[i + window_size];
            const weight = similarity_score * (1 / (len - i + 10));
            
            if (next_result === 'T') t_weight += weight;
            else x_weight += weight;
            
            total_similarity += similarity_score;
        }
    }
    
    if (total_similarity > 2.0) {
        const total_weight = t_weight + x_weight;
        if (total_weight > 0) {
            const ratio = Math.abs(t_weight - x_weight) / total_weight;
            if (ratio > 0.6) {
                return t_weight > x_weight ? 'T' : 'X';
            }
        }
    }
    
    return null;
}

// 7. THUẬT TOÁN SUPER BRIDGE ULTIMATE
function algog_super_bridge_predictor(history) {
    const runs = extract_features(history).runs;
    const tx = extract_features(history).tx;
    
    if (runs.length < 4) return null;
    
    const last_5_runs = runs.slice(-5);
    const recent_tx = tx.slice(-15);
    
    if (last_5_runs.length < 5) return null;
    
    const last_run = last_5_runs[4];
    const second_last = last_5_runs[3];
    
    // Pattern 1: Long run reversal
    if (last_run.len >= 4) {
        return last_run.val === 'T' ? 'X' : 'T';
    }
    
    // Pattern 2: Short run alternation
    if (last_run.len === 1 && second_last.len === 1) {
        let alternating_count = 0;
        for (let i = runs.length - 1; i >= Math.max(0, runs.length - 6); i--) {
            if (runs[i].len === 1) alternating_count++;
        }
        
        if (alternating_count >= 4) {
            return last_run.val === 'T' ? 'X' : 'T';
        }
    }
    
    // Pattern 3: Recent frequency imbalance
    const t_recent = recent_tx.filter(x => x === 'T').length;
    const x_recent = recent_tx.filter(x => x === 'X').length;
    
    if (t_recent > x_recent + 4) return 'X';
    if (x_recent > t_recent + 4) return 'T';
    
    return null;
}

// 8. THUẬT TOÁN ADAPTIVE MARKOV PRO
function algo_h_adaptive_markov(history) {
    const tx = extract_features(history).tx;
    if (tx.length < 25) return null;
    
    let best_pred = null;
    let best_confidence = 0;
    
    const max_order = Math.min(4, Math.floor(tx.length / 10));
    
    for (let order = 2; order <= max_order; order++) {
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
        
        if (counts && counts.t + counts.x >= 2) {
            const total = counts.t + counts.x;
            const confidence = Math.abs(counts.t - counts.x) / total;
            
            if (confidence > best_confidence) {
                best_confidence = confidence;
                best_pred = counts.t > counts.x ? 'T' : 'X';
            }
        }
    }
    
    // Ngưỡng confidence cao
    if (best_confidence > 0.7) {
        return best_pred;
    }
    
    return null;
}

// 9. THUẬT TOÁN NEURAL PATTERN RECOGNITION
function algoi_neural_pattern(history) {
    const tx = extract_features(history).tx;
    const totals = extract_features(history).totals;
    
    if (tx.length < 50) return null;
    
    // Feature extraction
    const recent_20 = tx.slice(-20);
    const recent_totals_20 = totals.slice(-20);
    
    const t_count_20 = recent_20.filter(x => x === 'T').length;
    const x_count_20 = recent_20.filter(x => x === 'X').length;
    const ratio_t_20 = t_count_20 / 20;
    
    const avg_total_20 = avg(recent_totals_20);
    
    const last_5 = tx.slice(-5);
    const last_5_pattern = last_5.join('');
    
    const first_10_avg = totals.length >= 20 ? avg(totals.slice(-20, -10)) : 0;
    const last_10_avg = totals.length >= 10 ? avg(totals.slice(-10)) : 0;
    const trend = last_10_avg - first_10_avg;
    
    // Neural-like decision making
    let t_score = 0, x_score = 0;
    
    // Rule 1: Strong recent bias
    if (ratio_t_20 > 0.7) x_score += 0.35;
    else if (ratio_t_20 < 0.3) t_score += 0.35;
    
    // Rule 2: Total average
    if (avg_total_20 > 12.5) x_score += 0.25;
    else if (avg_total_20 < 8.5) t_score += 0.25;
    
    // Rule 3: Pattern recognition
    const patterns = {
        'TTXTT': { t: 0.2, x: 0.8 },
        'TXTXT': { t: 0.3, x: 0.7 },
        'XXTXX': { t: 0.8, x: 0.2 },
        'XTXTX': { t: 0.7, x: 0.3 },
        'TTTTX': { t: 0.1, x: 0.9 },
        'XXXXT': { t: 0.9, x: 0.1 }
    };
    
    if (patterns[last_5_pattern]) {
        t_score += patterns[last_5_pattern].t * 0.2;
        x_score += patterns[last_5_pattern].x * 0.2;
    }
    
    // Rule 4: Trend analysis
    if (trend > 2.5) x_score += 0.15;
    else if (trend < -2.5) t_score += 0.15;
    
    const diff = Math.abs(t_score - x_score);
    if (diff > 0.25) {
        return t_score > x_score ? 'T' : 'X';
    }
    
    return null;
}

// 10. THUẬT TOÁN QUANTUM PREDICTOR
function algoj_quantum_predictor(history) {
    const tx = extract_features(history).tx;
    
    if (tx.length < 40) return null;
    
    // Quantum-inspired superposition of states
    let superposition_t = 0;
    let superposition_x = 0;
    
    // State 1: Frequency analysis
    const recent_30 = tx.slice(-30);
    const t_30 = recent_30.filter(x => x === 'T').length;
    const x_30 = recent_30.filter(x => x === 'X').length;
    
    if (t_30 > x_30 + 5) superposition_x += 0.3;
    else if (x_30 > t_30 + 5) superposition_t += 0.3;
    
    // State 2: Pattern entanglement
    const last_8 = tx.slice(-8);
    const pattern_weights = {
        't': 0,
        'x': 0
    };
    
    for (let i = 0; i <= tx.length - 9; i++) {
        const pattern = tx.slice(i, i + 8);
        if (pattern.join('') === last_8.join('')) {
            const next = tx[i + 8];
            pattern_weights[next.toLowerCase()]++;
        }
    }
    
    if (pattern_weights.t + pattern_weights.x >= 3) {
        const pattern_ratio = Math.abs(pattern_weights.t - pattern_weights.x) / (pattern_weights.t + pattern_weights.x);
        if (pattern_ratio > 0.6) {
            if (pattern_weights.t > pattern_weights.x) superposition_t += 0.25;
            else superposition_x += 0.25;
        }
    }
    
    // State 3: Entropy collapse
    const entropy_val = entropy(tx.slice(-20));
    if (entropy_val > 0.9) {
        // High entropy -> mean reversion
        if (tx[tx.length - 1] === 'T') superposition_x += 0.2;
        else superposition_t += 0.2;
    }
    
    // Quantum measurement
    const total_superposition = superposition_t + superposition_x;
    if (total_superposition > 0.5) {
        const confidence = Math.abs(superposition_t - superposition_x) / total_superposition;
        if (confidence > 0.6) {
            return superposition_t > superposition_x ? 'T' : 'X';
        }
    }
    
    return null;
}

// ==================== THUẬT TOÁN DỰ ĐOÁN 3 VỊ - PERFECT AI ====================

// THUẬT TOÁN DỰ ĐOÁN 3 VỊ - ULTIMATE PREDICTOR
function algod_score_predictor(history, tx_constraint) {
    const xiu_scores = [4, 5, 6, 7, 8, 9, 10];
    const tai_scores = [11, 12, 13, 14, 15, 16, 17];
    
    const available_scores = tx_constraint === 'T' ? tai_scores : xiu_scores;
    
    if (available_scores.length < 3) {
        const mid = Math.floor(available_scores.length / 2);
        return [
            available_scores[0] || 4,
            available_scores[mid] || 7,
            available_scores[available_scores.length - 1] || 10
        ].sort((a, b) => a - b);
    }
    
    // 1. PHÂN TÍCH TẦN SUẤT VỚI TIME DECAY
    const frequency_analysis = analyze_frequency(history, tx_constraint, available_scores);
    
    // 2. PHÂN TÍCH CHUỖI MARKOV
    const sequence_analysis = analyze_sequence(history, tx_constraint, available_scores);
    
    // 3. PHÂN TÍCH PHÂN BỐ
    const distribution_analysis = analyze_distribution(history, tx_constraint, available_scores);
    
    // 4. PHÂN TÍCH CLUSTER
    const cluster_analysis = analyze_cluster(history, tx_constraint, available_scores);
    
    // 5. PHÂN TÍCH GAP
    const gap_analysis = analyze_gap(history, tx_constraint, available_scores);
    
    // 6. KẾT HỢP TẤT CẢ PHÂN TÍCH
    const combined_scores = combine_analyses({
        frequency: frequency_analysis,
        sequence: sequence_analysis,
        distribution: distribution_analysis,
        cluster: cluster_analysis,
        gap: gap_analysis
    }, available_scores);
    
    // 7. CHỌN 3 VỊ ĐA DẠNG VÀ CHÍNH XÁC
    const selected_scores = select_optimal_scores(combined_scores, available_scores);
    
    // 8. ĐẢM BẢO COVERAGE ĐẦY ĐỦ
    return ensure_full_coverage(selected_scores, available_scores, tx_constraint);
}

function analyze_frequency(history, tx_constraint, available_scores) {
    const lookback = Math.min(history.length, 80);
    const decay_rate = 0.96;
    
    const score_weights = {};
    
    for (let i = Math.max(0, history.length - lookback); i < history.length - 1; i++) {
        if (history[i].tx === tx_constraint) {
            const next_score = history[i + 1].total;
            if (available_scores.includes(next_score)) {
                const age = history.length - 1 - i;
                const time_weight = Math.pow(decay_rate, age);
                score_weights[next_score] = (score_weights[next_score] || 0) + time_weight;
            }
        }
    }
    
    // Normalize
    const total_weight = Object.values(score_weights).reduce((a, b) => a + b, 0);
    const predictions = {};
    
    if (total_weight > 0) {
        for (let score in score_weights) {
            predictions[parseInt(score)] = score_weights[score] / total_weight;
        }
    }
    
    // Đảm bảo tất cả scores đều có giá trị
    for (let score of available_scores) {
        if (!predictions[score]) {
            predictions[score] = 0.1;
        }
    }
    
    return predictions;
}

function analyze_sequence(history, tx_constraint, available_scores) {
    const predictions = {};
    
    // Markov chain bậc 2
    const transitions = {};
    for (let i = 2; i < history.length - 1; i++) {
        if (history[i-2].tx === tx_constraint && 
            history[i-1].tx === tx_constraint && 
            history[i].tx === tx_constraint) {
            
            const state = `${history[i-2].total},${history[i-1].total}`;
            const next_score = history[i].total;
            
            if (!transitions[state]) {
                transitions[state] = {};
            }
            transitions[state][next_score] = (transitions[state][next_score] || 0) + 1;
        }
    }
    
    // Dự đoán từ 2 state gần nhất
    if (history.length >= 2) {
        const last_state = `${history[history.length-2]?.total || 0},${history[history.length-1]?.total || 0}`;
        
        if (transitions[last_state]) {
            const total = Object.values(transitions[last_state]).reduce((a, b) => a + b, 0);
            for (let score_str in transitions[last_state]) {
                const score = parseInt(score_str);
                if (available_scores.includes(score)) {
                    predictions[score] = transitions[last_state][score_str] / total;
                }
            }
        }
    }
    
    // Fallback: sử dụng phân phối đều
    for (let score of available_scores) {
        if (!predictions[score]) {
            predictions[score] = 1.0 / available_scores.length;
        }
    }
    
    return predictions;
}

function analyze_distribution(history, tx_constraint, available_scores) {
    const all_scores = history.filter(h => h.tx === tx_constraint).map(h => h.total);
    
    if (all_scores.length < 5) {
        const predictions = {};
        for (let score of available_scores) {
            predictions[score] = 1.0 / available_scores.length;
        }
        return predictions;
    }
    
    // Tính histogram
    const histogram = {};
    for (let score of all_scores) {
        histogram[score] = (histogram[score] || 0) + 1;
    }
    
    // Normalize
    const total = all_scores.length;
    const predictions = {};
    
    for (let score of available_scores) {
        const count = histogram[score] || 0;
        predictions[score] = (count + 1) / (total + available_scores.length);
    }
    
    return predictions;
}

function analyze_cluster(history, tx_constraint, available_scores) {
    const recent_scores = history
        .slice(-30)
        .filter(h => h.tx === tx_constraint)
        .map(h => h.total);
    
    if (recent_scores.length < 3) {
        const predictions = {};
        for (let score of available_scores) {
            predictions[score] = 1.0 / available_scores.length;
        }
        return predictions;
    }
    
    // Phát hiện cluster
    const sorted_scores = [...recent_scores].sort((a, b) => a - b);
    const clusters = [];
    let current_cluster = [];
    
    for (let i = 0; i < sorted_scores.length; i++) {
        if (current_cluster.length === 0 || 
            sorted_scores[i] - current_cluster[current_cluster.length - 1] <= 2) {
            current_cluster.push(sorted_scores[i]);
        } else {
            if (current_cluster.length >= 2) {
                clusters.push({
                    scores: [...new Set(current_cluster)],
                    size: current_cluster.length
                });
            }
            current_cluster = [sorted_scores[i]];
        }
    }
    
    if (current_cluster.length >= 2) {
        clusters.push({
            scores: [...new Set(current_cluster)],
            size: current_cluster.length
        });
    }
    
    const predictions = {};
    
    // Ưu tiên điểm số trong cluster phổ biến
    for (let score of available_scores) {
        let cluster_bonus = 1.0;
        
        for (let cluster of clusters) {
            if (cluster.scores.includes(score)) {
                cluster_bonus += cluster.size * 0.1;
            }
        }
        
        predictions[score] = cluster_bonus;
    }
    
    // Normalize
    const total = Object.values(predictions).reduce((a, b) => a + b, 0);
    for (let score in predictions) {
        predictions[score] /= total;
    }
    
    return predictions;
}

function analyze_gap(history, tx_constraint, available_scores) {
    const predictions = {};
    
    // Tìm các điểm số ít xuất hiện gần đây
    for (let score of available_scores) {
        let last_seen = -1;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].tx === tx_constraint && history[i].total === score) {
                last_seen = i;
                break;
            }
        }
        
        if (last_seen === -1) {
            predictions[score] = 0.7; // Chưa bao giờ xuất hiện
        } else {
            const gap = history.length - last_seen;
            predictions[score] = Math.min(1.0, gap / 15);
        }
    }
    
    // Normalize
    const total = Object.values(predictions).reduce((a, b) => a + b, 0);
    for (let score in predictions) {
        predictions[score] /= total;
    }
    
    return predictions;
}

function combine_analyses(analyses, available_scores) {
    const combined = {};
    
    // Trọng số cho từng phân tích
    const weights = {
        frequency: 0.25,
        sequence: 0.20,
        distribution: 0.20,
        cluster: 0.20,
        gap: 0.15
    };
    
    for (let score of available_scores) {
        combined[score] = 0;
        
        for (let analysis in weights) {
            if (analyses[analysis] && analyses[analysis][score]) {
                combined[score] += analyses[analysis][score] * weights[analysis];
            }
        }
    }
    
    return combined;
}

function select_optimal_scores(combined_scores, available_scores) {
    const scoreArray = Object.entries(combined_scores)
        .map(([score, prob]) => ({ score: parseInt(score), probability: prob }))
        .sort((a, b) => b.probability - a.probability);
    
    if (scoreArray.length === 0) {
        return get_default_selection(available_scores);
    }
    
    // Chiến lược chọn đa dạng: chọn từ các phần khác nhau của phạm vi
    const selected = [];
    const range_min = Math.min(...available_scores);
    const range_max = Math.max(...available_scores);
    const range_mid = Math.floor((range_min + range_max) / 2);
    
    // Chọn điểm cao nhất
    if (scoreArray.length > 0) {
        selected.push(scoreArray[0].score);
    }
    
    // Tìm điểm ở phần thấp
    let best_low = null;
    let best_low_prob = -1;
    
    // Tìm điểm ở phần cao
    let best_high = null;
    let best_high_prob = -1;
    
    // Tìm điểm ở phần giữa
    let best_mid = null;
    let best_mid_prob = -1;
    
    for (let item of scoreArray) {
        const score = item.score;
        const prob = item.probability;
        
        if (score <= range_mid - 1 && prob > best_low_prob && !selected.includes(score)) {
            best_low = score;
            best_low_prob = prob;
        }
        
        if (score >= range_mid + 1 && prob > best_high_prob && !selected.includes(score)) {
            best_high = score;
            best_high_prob = prob;
        }
        
        if (Math.abs(score - range_mid) <= 1 && prob > best_mid_prob && !selected.includes(score)) {
            best_mid = score;
            best_mid_prob = prob;
        }
    }
    
    // Ưu tiên: thấp, cao, giữa
    if (best_low !== null && selected.length < 3) {
        selected.push(best_low);
    }
    
    if (best_high !== null && selected.length < 3) {
        selected.push(best_high);
    }
    
    if (best_mid !== null && selected.length < 3) {
        selected.push(best_mid);
    }
    
    // Nếu vẫn thiếu, thêm từ các điểm có xác suất cao tiếp theo
    if (selected.length < 3) {
        for (let item of scoreArray) {
            if (!selected.includes(item.score)) {
                selected.push(item.score);
                if (selected.length >= 3) break;
            }
        }
    }
    
    // Fallback cuối cùng
    if (selected.length < 3) {
        const remaining = available_scores.filter(s => !selected.includes(s));
        while (selected.length < 3 && remaining.length > 0) {
            selected.push(remaining.shift());
        }
    }
    
    return selected;
}

function get_default_selection(available_scores) {
    const len = available_scores.length;
    if (len >= 3) {
        return [
            available_scores[0],
            available_scores[Math.floor(len / 2)],
            available_scores[len - 1]
        ];
    }
    return available_scores.slice(0, 3);
}

function ensure_full_coverage(selected, available_scores, tx_constraint) {
    const key = tx_constraint; // 'T' hoặc 'X'
    
    // Lấy danh sách dự đoán gần đây từ memory
    const recent = prediction_memory[key] || [];
    
    // Thêm dự đoán hiện tại
    recent.push(selected);
    if (recent.length > 10) {
        recent.shift();
    }
    
    // Cập nhật memory
    prediction_memory[key] = recent;
    
    // Check coverage
    const all_selected = recent.flat();
    const coverage = {};
    
    for (let score of available_scores) {
        coverage[score] = all_selected.filter(s => s === score).length;
    }
    
    // Tìm các điểm số ít được dự đoán
    const min_coverage = Math.min(...Object.values(coverage));
    const underrepresented = available_scores.filter(s => coverage[s] === min_coverage);
    
    // Nếu có điểm số ít được dự đoán, đôi khi thay thế
    if (underrepresented.length > 0 && Math.random() < 0.3) {
        const replace_idx = Math.floor(Math.random() * selected.length);
        const new_score = underrepresented[Math.floor(Math.random() * underrepresented.length)];
        
        if (!selected.includes(new_score)) {
            selected[replace_idx] = new_score;
        }
    }
    
    return selected.sort((a, b) => a - b);
}

// --- DANH SÁCH THUẬT TOÁN KẾT HỢP ---
const all_algs = [{
    id: 'algo5_freq_rebalance',
    fn: algo5_freq_rebalance
}, {
    id: 'a_markov',
    fn: algoa_markov
}, {
    id: 'b_ngram',
    fn: algob_ngram
}, {
    id: 's_neo_pattern',
    fn: algos_neo_pattern
}, {
    id: 'f_super_deep_analysis', 
    fn: algof_super_deep_analysis
}, {
    id: 'e_transformer', 
    fn: algoe_transformer
}, {
    id: 'g_super_bridge_predictor', 
    fn: algog_super_bridge_predictor
}, {
    id: 'h_adaptive_markov', 
    fn: algo_h_adaptive_markov
}, {
    id: 'i_neural_pattern', 
    fn: algoi_neural_pattern
}, {
    id: 'j_quantum_predictor', 
    fn: algoj_quantum_predictor
}];

// --- ENSEMBLE CLASSIFIER ---
class SeiuEnsemble {
    constructor(algorithms, opts = {}) { 
        this.algs = algorithms;
        this.weights = {};
        this.ema_alpha = opts.ema_alpha ?? 0.1;
        this.min_weight = opts.min_weight ?? 0.001;
        this.history_window = opts.history_window ?? 500;
        this.performance_history = {};
        
        for (let a of algorithms) {
            this.weights[a.id] = 1;
            this.performance_history[a.id] = { correct: 0, total: 0 };
        }
    }
    
    fit_initial(history) {
        const window = last_n(history.filter(h => h.tx !== 'B'), this.history_window);
        if (window.length < 30) return;
        
        console.log(`🧠 Đang huấn luyện AI với ${window.length} phiên lịch sử...`);
        
        for (let i = 20; i < window.length; i++) {
            const prefix = window.slice(0, i);
            const actual = window[i].tx;
            
            for (let a of this.algs) {
                const pred = a.fn(prefix);
                if (pred) {
                    this.performance_history[a.id].total++;
                    if (pred === actual) {
                        this.performance_history[a.id].correct++;
                    }
                }
            }
        }
        
        // Tính trọng số dựa trên độ chính xác
        let total_weight = 0;
        for (let a of this.algs) {
            const perf = this.performance_history[a.id];
            const accuracy = perf.total > 0 ? perf.correct / perf.total : 0.5;
            const adjusted_accuracy = Math.pow(accuracy, 1.5);
            this.weights[a.id] = Math.max(this.min_weight, adjusted_accuracy);
            total_weight += this.weights[a.id];
        }
        
        // Chuẩn hóa trọng số
        for (let id in this.weights) {
            this.weights[id] /= total_weight;
        }
        
        console.log(`⚖️ Đã khởi tạo trọng số cho ${Object.keys(this.weights).length} thuật toán.`);
        
        // Log performance
        for (let a of this.algs) {
            const perf = this.performance_history[a.id];
            if (perf.total > 0) {
                const acc = (perf.correct / perf.total * 100).toFixed(1);
                console.log(`   ${a.id}: ${acc}% (${perf.correct}/${perf.total})`);
            }
        }
    }

    update_with_outcome(history_prefix, actual_tx) {
        if (actual_tx === 'B') return; 
        
        for (let a of this.algs) {
            const pred = a.fn(history_prefix);
            
            if (pred) {
                const perf = this.performance_history[a.id];
                perf.total++;
                
                if (pred === actual_tx) {
                    perf.correct++;
                }
                
                // Cập nhật trọng số
                const recent_accuracy = perf.correct / perf.total;
                const current_weight = this.weights[a.id] || this.min_weight;
                
                const learning_rate = 0.2;
                const target_weight = Math.min(1, Math.max(0.01, recent_accuracy));
                const new_weight = learning_rate * target_weight + (1 - learning_rate) * current_weight;
                
                this.weights[a.id] = Math.max(this.min_weight, new_weight);
            }
        }

        // Chuẩn hóa trọng số
        const total = Object.values(this.weights).reduce((a, b) => a + b, 0) || 1;
        for (let id in this.weights) {
            this.weights[id] /= total;
        }
    }

    predict(history) {
        const votes = {};
        
        for (let a of this.algs) {
            const pred = a.fn(history);
            if (!pred) continue;
            
            votes[pred] = (votes[pred] || 0) + (this.weights[a.id] || 0);
        }

        let best, confidence;

        if (!votes['T'] && !votes['X']) {
            best = algo5_freq_rebalance(history) || 'T';
            confidence = 0.5;
        } else {
            const result = majority(votes);
            best = result.key;
            const total = Object.values(votes).reduce((a, b) => a + b, 0);
            confidence = Math.min(0.99, Math.max(0.51, total > 0 ? result.val / total : 0.51));
        }

        // Gọi hàm dự đoán 3 vị
        const score_prediction = algod_score_predictor(history, best);

        return {
            prediction: best === 'T' ? 'tài' : 'xỉu',
            confidence,
            raw_prediction: best,
            score_prediction
        };
    }
}

// --- MANAGER CLASS ---
class SeiuManager {
    constructor(opts = {}) {
        this.history = [];
        this.ensemble = new SeiuEnsemble(all_algs, {
            ema_alpha: opts.ema_alpha ?? 0.1,
            history_window: opts.history_window ?? 500
        });
        this.current_prediction = null;
        this.accuracy_stats = { correct: 0, total: 0 };
    }
    
    calculate_initial_stats() {
        const min_start = 20;
        const filtered_history = this.history.filter(h => h.tx !== 'B');

        if (filtered_history.length < min_start) return;
        
        for (let i = min_start; i < filtered_history.length; i++) {
            const history_prefix = filtered_history.slice(0, i);
            const actual_tx = filtered_history[i].tx;
            this.ensemble.update_with_outcome(history_prefix, actual_tx);
        }
    }

    load_initial(lines) {
        this.history = lines;
        this.ensemble.fit_initial(this.history);
        this.calculate_initial_stats();
        this.current_prediction = this.get_prediction();
        
        console.log("📦 Đã tải lịch sử. Hệ thống sẵn sàng.");
        const next_session = this.history.at(-1) ? this.history.at(-1).session + 1 : 'n/a';
        const score_pred_str = this.current_prediction.score_prediction.join('-');
        
        console.log(`🔮 Dự đoán phiên tiếp theo (${next_session}): ${this.current_prediction.prediction} (tỷ lệ: ${(this.current_prediction.confidence * 100).toFixed(0)}%). Vị (tổng điểm): [${score_pred_str}]`);
    }

    push_record(record) {
        this.history.push(record);

        const prefix = this.history.slice(0, -1).filter(h => h.tx !== 'B');
        if (prefix.length >= 10) {
            this.ensemble.update_with_outcome(prefix, record.tx);
            
            // Update accuracy stats
            const last_prediction = this.current_prediction;
            if (last_prediction) {
                this.accuracy_stats.total++;
                const predicted_tx = last_prediction.raw_prediction === 'T' ? 'TAI' : 'XIU';
                if (predicted_tx === record.result) {
                    this.accuracy_stats.correct++;
                }
                
                // Log accuracy every 50 predictions
                if (this.accuracy_stats.total % 50 === 0) {
                    const accuracy = (this.accuracy_stats.correct / this.accuracy_stats.total * 100).toFixed(1);
                    console.log(`🎯 Độ chính xác: ${accuracy}% (${this.accuracy_stats.correct}/${this.accuracy_stats.total})`);
                }
            }
        }
        
        this.current_prediction = this.get_prediction();
        const score_pred_str = this.current_prediction.score_prediction.join('-');
        
        console.log(`📥 Phiên mới ${record.session} → ${record.result.toLowerCase()}. Dự đoán phiên ${record.session + 1} là: ${this.current_prediction.prediction}. Vị (tổng điểm): [${score_pred_str}]`);
    }

    get_prediction() {
        return this.ensemble.predict(this.history);
    }
}

const seiu_manager = new SeiuManager();

// --- API SERVER ---
const app = fastify({
    logger: true
});
await app.register(cors, {
    origin: "*"
});

async function fetch_and_process_history() {
    try {
        console.log(`🔄 Đang lấy dữ liệu từ API...`);
        const response = await fetch(api_url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const new_history = parse_lines(data); 
        
        if (new_history.length === 0) {
            console.log("⚠️ Không có dữ liệu lịch sử từ API.");
            return;
        }

        const last_session_in_history = new_history.at(-1);

        if (!current_session_id) {
            seiu_manager.load_initial(new_history);
            txh_history = new_history;
            current_session_id = last_session_in_history.session;
            console.log(`✅ Lần đầu tải ${new_history.length} phiên. Phiên hiện tại: ${current_session_id}`);
        } else if (last_session_in_history.session > current_session_id) {
            const new_records = new_history.filter(r => r.session > current_session_id);
            
            if (new_records.length > 0) {
                for (let record of new_records) {
                    seiu_manager.push_record(record);
                    txh_history.push(record);
                }
                
                // Giữ lại 500 phiên gần nhất
                if (txh_history.length > 500) {
                    txh_history = txh_history.slice(txh_history.length - 500);
                }
                
                current_session_id = last_session_in_history.session;
                console.log(`🆕 Đã cập nhật ${new_records.length} phiên mới. Phiên cuối: ${current_session_id}`);
            } else {
                console.log(`🔄 Không có phiên mới. Phiên cuối: ${current_session_id}`);
            }
        } else {
            console.log(`🔄 Dữ liệu không thay đổi. Phiên cuối: ${current_session_id}`);
        }

    } catch (e) {
        console.error("❌ Lỗi khi lấy hoặc xử lý lịch sử:", e.message);
    }
}

// Lấy dữ liệu lần đầu
fetch_and_process_history();

// Thiết lập việc lấy dữ liệu định kỳ
clearInterval(fetch_interval);
fetch_interval = setInterval(fetch_and_process_history, 5000); 
console.log(`🔄 Đang thiết lập fetch API mỗi 5 giây tại URL: ${api_url}`);

// GET /api/sicbo/sunwin
app.get("/api/sicbo/sunwin", async () => {
    const last_result = txh_history.at(-1) || null; 
    const current_prediction = seiu_manager.current_prediction;
    
    const score_pred_str = current_prediction?.score_prediction ? current_prediction.score_prediction.join('-') : 'chưa có';
    
    if (!last_result || !current_prediction) {
        return {
            "id": "GiaThinhzZz Sicbo Sunwin",
            "phien_truoc": null,
            "xuc_xac1": null,
            "xuc_xac2": null,
            "xuc_xac3": null,
            "tong": null,
            "ket_qua": "đang chờ dữ liệu",
            "phien_hien_tai": current_session_id ? current_session_id + 1 : null,
            "du_doan": "chưa có",
            "du_doan_vi": score_pred_str,
            "do_tin_cay": "0%"
        };
    }

    return {
        "id": "GiaThinhzZz Sicbo Sunwin",
        "phien_truoc": last_result.session,
        "xuc_xac1": last_result.dice[0],
        "xuc_xac2": last_result.dice[1],
        "xuc_xac3": last_result.dice[2],
        "tong": last_result.total,
        "ket_qua": last_result.result.toLowerCase(),
        "phien_hien_tai": last_result.session + 1,
        "du_doan": current_prediction.prediction,
        "du_doan_vi": score_pred_str, 
        "do_tin_cay": `${(current_prediction.confidence * 100).toFixed(0)}%`,
    };
});

// GET /api/sicbo/history
app.get("/api/sicbo/history", async () => { 
    if (!txh_history.length) return {
        message: "không có dữ liệu lịch sử."
    };
    const reversed_history = [...txh_history].sort((a, b) => b.session - a.session);
    
    return reversed_history.map((i) => ({
        session: i.session,
        dice: i.dice,
        total: i.total,
        result: i.result.toLowerCase(),
        tx_label: i.tx.toLowerCase(),
    }));
});

// GET /api/sicbo/stats
app.get("/api/sicbo/stats", async () => {
    const accuracy = seiu_manager.accuracy_stats.total > 0 ? 
        (seiu_manager.accuracy_stats.correct / seiu_manager.accuracy_stats.total * 100).toFixed(1) : "0.0";
    
    return {
        "tong_phien": txh_history.length,
        "phien_hien_tai": current_session_id,
        "do_chinh_xac": `${accuracy}%`,
        "tong_du_doan": seiu_manager.accuracy_stats.total,
        "du_doan_dung": seiu_manager.accuracy_stats.correct,
        "so_thuat_toan": all_algs.length,
        "pham_vi_vi": {
            "xiu": [4, 5, 6, 7, 8, 9, 10],
            "tai": [11, 12, 13, 14, 15, 16, 17]
        },
        "memory": {
            "tai_predictions": prediction_memory.T?.length || 0,
            "xiu_predictions": prediction_memory.X?.length || 0
        }
    };
});

// GET /
app.get("/", async () => { 
    return {
        status: "ok",
        msg: "server chạy thành công 🚀",
        endpoints: {
            "/api/sicbo/sunwin": "dự đoán tài xỉu và 3 vị",
            "/api/sicbo/history": "lịch sử kết quả",
            "/api/sicbo/stats": "thống kê hệ thống"
        },
        thong_tin: {
            "so_thuat_toan": all_algs.length,
            "phien_hien_tai": current_session_id,
            "tong_phien_luu": txh_history.length,
            "pham_vi_vi_xiu": "4,5,6,7,8,9,10",
            "pham_vi_vi_tai": "11,12,13,14,15,16,17"
        }
    };
});

// --- SERVER START ---
const start = async () => {
    try {
        await app.listen({
            port: port,
            host: "0.0.0.0"
        });
    } catch (err) {
        const error_msg = `
================= SERVER ERROR =================
time: ${new Date().toISOString()}
error: ${err.message}
stack: ${err.stack}
=================================================
`;
        console.error(error_msg);
        
        // Ghi log vào file
        const log_file = path.join(__dirname, "server-error.log");
        fs.writeFileSync(log_file, error_msg, {
            encoding: "utf8",
            flag: "a+"
        });
        process.exit(1);
    }

    let public_ip = "0.0.0.0";
    try {
        const res = await fetch("https://ifconfig.me/ip");
        public_ip = (await res.text()).trim();
    } catch (e) {
        console.error("❌ Lỗi lấy public IP:", e.message);
    }

    console.log("\n" + "=".repeat(60));
    console.log("🚀 SICBO SUNWIN AI SERVER ĐÃ CHẠY THÀNH CÔNG!");
    console.log("=".repeat(60));
    console.log(`   ➜ Local:   http://localhost:${port}/`);
    console.log(`   ➜ Network: http://${public_ip}:${port}/`);
    console.log("\n📌 CÁC API ENDPOINTS:");
    console.log(`   ➜ GET /api/sicbo/sunwin   → Dự đoán Tài/Xỉu + 3 Vị`);
    console.log(`   ➜ GET /api/sicbo/history  → Lịch sử 500 phiên gần nhất`);
    console.log(`   ➜ GET /api/sicbo/stats    → Thống kê độ chính xác`);
    
    console.log("\n🎯 THÔNG TIN HỆ THỐNG AI:");
    console.log(`   Số thuật toán: ${all_algs.length}`);
    console.log(`   Phạm vi vị Xỉu: 4,5,6,7,8,9,10 (ĐẦY ĐỦ)`);
    console.log(`   Phạm vi vị Tài: 11,12,13,14,15,16,17 (ĐẦY ĐỦ)`);
    console.log(`   Đảm bảo đầy đủ tất cả các vị!`);
    console.log("=".repeat(60));
    console.log("\n🤖 Hệ thống đang chạy... Chờ dữ liệu từ API...");
};

start();
