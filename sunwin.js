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
let prediction_memory = { T: [], X: [] };

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

// ==================== THUẬT TOÁN DỰ ĐOÁN 3 VỊ - ULTIMATE PREDICTOR (GIỮ NGUYÊN) ====================

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
    
    const frequency_analysis = analyze_frequency(history, tx_constraint, available_scores);
    const sequence_analysis = analyze_sequence(history, tx_constraint, available_scores);
    const distribution_analysis = analyze_distribution(history, tx_constraint, available_scores);
    const cluster_analysis = analyze_cluster(history, tx_constraint, available_scores);
    const gap_analysis = analyze_gap(history, tx_constraint, available_scores);
    
    const combined_scores = combine_analyses({
        frequency: frequency_analysis,
        sequence: sequence_analysis,
        distribution: distribution_analysis,
        cluster: cluster_analysis,
        gap: gap_analysis
    }, available_scores);
    
    const selected_scores = select_optimal_scores(combined_scores, available_scores);
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
    
    const total_weight = Object.values(score_weights).reduce((a, b) => a + b, 0);
    const predictions = {};
    
    if (total_weight > 0) {
        for (let score in score_weights) {
            predictions[parseInt(score)] = score_weights[score] / total_weight;
        }
    }
    
    for (let score of available_scores) {
        if (!predictions[score]) {
            predictions[score] = 0.1;
        }
    }
    
    return predictions;
}

function analyze_sequence(history, tx_constraint, available_scores) {
    const predictions = {};
    
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
    
    const histogram = {};
    for (let score of all_scores) {
        histogram[score] = (histogram[score] || 0) + 1;
    }
    
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
    
    for (let score of available_scores) {
        let cluster_bonus = 1.0;
        for (let cluster of clusters) {
            if (cluster.scores.includes(score)) {
                cluster_bonus += cluster.size * 0.1;
            }
        }
        predictions[score] = cluster_bonus;
    }
    
    const total = Object.values(predictions).reduce((a, b) => a + b, 0);
    for (let score in predictions) {
        predictions[score] /= total;
    }
    
    return predictions;
}

function analyze_gap(history, tx_constraint, available_scores) {
    const predictions = {};
    
    for (let score of available_scores) {
        let last_seen = -1;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].tx === tx_constraint && history[i].total === score) {
                last_seen = i;
                break;
            }
        }
        
        if (last_seen === -1) {
            predictions[score] = 0.7;
        } else {
            const gap = history.length - last_seen;
            predictions[score] = Math.min(1.0, gap / 15);
        }
    }
    
    const total = Object.values(predictions).reduce((a, b) => a + b, 0);
    for (let score in predictions) {
        predictions[score] /= total;
    }
    
    return predictions;
}

function combine_analyses(analyses, available_scores) {
    const combined = {};
    const weights = { frequency: 0.25, sequence: 0.20, distribution: 0.20, cluster: 0.20, gap: 0.15 };
    
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
    
    if (scoreArray.length === 0) return get_default_selection(available_scores);
    
    const selected = [];
    const range_min = Math.min(...available_scores);
    const range_max = Math.max(...available_scores);
    const range_mid = Math.floor((range_min + range_max) / 2);
    
    if (scoreArray.length > 0) selected.push(scoreArray[0].score);
    
    let best_low = null, best_low_prob = -1;
    let best_high = null, best_high_prob = -1;
    let best_mid = null, best_mid_prob = -1;
    
    for (let item of scoreArray) {
        const score = item.score;
        const prob = item.probability;
        
        if (score <= range_mid - 1 && prob > best_low_prob && !selected.includes(score)) {
            best_low = score; best_low_prob = prob;
        }
        if (score >= range_mid + 1 && prob > best_high_prob && !selected.includes(score)) {
            best_high = score; best_high_prob = prob;
        }
        if (Math.abs(score - range_mid) <= 1 && prob > best_mid_prob && !selected.includes(score)) {
            best_mid = score; best_mid_prob = prob;
        }
    }
    
    if (best_low !== null && selected.length < 3) selected.push(best_low);
    if (best_high !== null && selected.length < 3) selected.push(best_high);
    if (best_mid !== null && selected.length < 3) selected.push(best_mid);
    
    if (selected.length < 3) {
        for (let item of scoreArray) {
            if (!selected.includes(item.score)) {
                selected.push(item.score);
                if (selected.length >= 3) break;
            }
        }
    }
    
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
        return [available_scores[0], available_scores[Math.floor(len / 2)], available_scores[len - 1]];
    }
    return available_scores.slice(0, 3);
}

function ensure_full_coverage(selected, available_scores, tx_constraint) {
    const key = tx_constraint;
    const recent = prediction_memory[key] || [];
    recent.push(selected);
    if (recent.length > 10) recent.shift();
    prediction_memory[key] = recent;
    
    const all_selected = recent.flat();
    const coverage = {};
    for (let score of available_scores) {
        coverage[score] = all_selected.filter(s => s === score).length;
    }
    
    const min_coverage = Math.min(...Object.values(coverage));
    const underrepresented = available_scores.filter(s => coverage[s] === min_coverage);
    
    if (underrepresented.length > 0 && Math.random() < 0.3) {
        const replace_idx = Math.floor(Math.random() * selected.length);
        const new_score = underrepresented[Math.floor(Math.random() * underrepresented.length)];
        if (!selected.includes(new_score)) selected[replace_idx] = new_score;
    }
    
    return selected.sort((a, b) => a - b);
}

// ==================== 6 MODEL CẦU MỚI ====================

function engineAdvancedPredict(validItems) {
    let duDoan = 'Tài';
    let conf = 68;

    if (!validItems || validItems.length < 2) {
        return { duDoan, conf: 55 };
    }

    const h = validItems.slice(0, 12);
    const n = h.length;

    const results = h.map(item => {
        const kq = String(item.result || '').toUpperCase();
        return (kq === 'TAI' || kq === 'TÀI') ? 1 : 0;
    });

    let weightedTai = 0, weightedTotal = 0;
    for (let i = 0; i < n; i++) {
        const weight = Math.pow(0.85, i);
        weightedTai += results[i] * weight;
        weightedTotal += weight;
    }
    const weightedRatio = weightedTotal > 0 ? weightedTai / weightedTotal : 0.5;

    let currentStreak = 1;
    const lastResult = results[0];
    for (let i = 1; i < n; i++) {
        if (results[i] === lastResult) currentStreak++;
        else break;
    }

    let isAlternating = true;
    if (n >= 4) {
        for (let i = 0; i < Math.min(6, n - 1); i++) {
            if (results[i] === results[i + 1]) { isAlternating = false; break; }
        }
    } else { isAlternating = false; }

    const shortWindow = results.slice(0, Math.min(5, n));
    const shortTai = shortWindow.filter(r => r === 1).length;
    const shortRatio = shortTai / shortWindow.length;

    const longTai = results.filter(r => r === 1).length;
    const longRatio = longTai / n;

    let scoreTai = 0;

    if (currentStreak >= 4) {
        const reverseStrength = Math.min(0.35 + (currentStreak - 4) * 0.08, 0.55);
        if (lastResult === 1) scoreTai -= reverseStrength;
        else scoreTai += reverseStrength;
    } else if (currentStreak >= 2) {
        const mom = 0.18 + (currentStreak - 2) * 0.07;
        if (lastResult === 1) scoreTai += mom;
        else scoreTai -= mom;
    }

    scoreTai += (weightedRatio - 0.5) * 0.9;
    scoreTai += (shortRatio - 0.5) * 0.55;

    if (isAlternating) {
        if (lastResult === 1) scoreTai -= 0.28;
        else scoreTai += 0.28;
    }

    scoreTai += (longRatio - 0.5) * 0.25;

    if (scoreTai > 0.08) duDoan = 'Tài';
    else if (scoreTai < -0.08) duDoan = 'Xỉu';
    else duDoan = lastResult === 1 ? 'Xỉu' : 'Tài';

    const signalStrength = Math.min(Math.abs(scoreTai) * 1.8, 1);
    conf = Math.round(58 + signalStrength * 32);

    if (currentStreak >= 4) conf = Math.min(conf + 6, 92);
    if (isAlternating && n >= 5) conf = Math.min(conf + 4, 90);
    if (n < 5) conf = Math.min(conf, 68);
    conf = Math.max(52, Math.min(92, conf));

    return { duDoan, conf };
}

// --- MANAGER CLASS ---
class SeiuManager {
    constructor() {
        this.history = [];
        this.current_prediction = null;
        this.accuracy_stats = { correct: 0, total: 0 };
    }
    
    calculate_initial_stats() {
        if (this.history.length < 20) return;
        for (let i = 20; i < this.history.length; i++) {
            const history_prefix = this.history.slice(0, i);
            const actual_tx = this.history[i].tx;
            
            if (actual_tx === 'B') continue;
            const pred = engineAdvancedPredict([...history_prefix].reverse());
            const predicted_tx = pred.duDoan === 'Tài' ? 'TAI' : 'XIU';
            if (predicted_tx === actual_tx) {
                this.accuracy_stats.correct++;
            }
            this.accuracy_stats.total++;
        }
    }

    load_initial(lines) {
        this.history = lines;
        this.calculate_initial_stats();
        this.current_prediction = this.get_prediction();
        
        console.log("📦 Đã tải lịch sử. Hệ thống sẵn sàng.");
    }

    push_record(record) {
        this.history.push(record);
        
        const prefix = this.history.slice(0, -1);
        if (prefix.length >= 10 && record.tx !== 'B') {
            const last_prediction = this.current_prediction;
            if (last_prediction) {
                const predicted_tx = last_prediction.raw_prediction === 'T' ? 'TAI' : 'XIU';
                if (predicted_tx === record.result) {
                    this.accuracy_stats.correct++;
                }
                this.accuracy_stats.total++;
            }
        }
        
        this.current_prediction = this.get_prediction();
        
        console.log(`📥 Phiên mới ${record.session} → ${record.result.toLowerCase()}. Dự đoán phiên ${record.session + 1} là: ${this.current_prediction.prediction}. Vị (tổng điểm): [${this.current_prediction.score_prediction.join('-')}]`);
    }

    get_prediction() {
        const reversed_history = [...this.history].reverse();
        const pred6 = engineAdvancedPredict(reversed_history);
        const raw_pred = pred6.duDoan === 'Tài' ? 'T' : 'X';
        const score_pred = algod_score_predictor(this.history, raw_pred);
        
        return {
            prediction: pred6.duDoan.toLowerCase(),
            confidence: pred6.conf / 100,
            raw_prediction: raw_pred,
            score_prediction: score_pred
        };
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
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const new_history = parse_lines(data); 
        
        if (new_history.length === 0) return;

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
                if (txh_history.length > 500) txh_history = txh_history.slice(txh_history.length - 500);
                current_session_id = last_session_in_history.session;
                console.log(`🆕 Đã cập nhật ${new_records.length} phiên mới.`);
            }
        }

    } catch (e) {
        console.error("❌ Lỗi khi lấy hoặc xử lý lịch sử:", e.message);
    }
}

fetch_and_process_history();
clearInterval(fetch_interval);
fetch_interval = setInterval(fetch_and_process_history, 5000); 

// GET /api/sicbo/sunwin
app.get("/api/sicbo/sunwin", async () => {
    const last_result = txh_history.at(-1) || null; 
    const current_prediction = seiu_manager.current_prediction;
    
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
            "du_doan_vi": "chưa có",
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
        "du_doan_vi": current_prediction.score_prediction.join('-'), 
        "do_tin_cay": `${(current_prediction.confidence * 100).toFixed(0)}%`,
    };
});

// GET /api/sicbo/history
app.get("/api/sicbo/history", async () => { 
    if (!txh_history.length) return { message: "không có dữ liệu lịch sử." };
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
        "so_thuat_toan": "6 Model V2",
        "pham_vi_vi": {
            "xiu": [4, 5, 6, 7, 8, 9, 10],
            "tai": [11, 12, 13, 14, 15, 16, 17]
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
            "so_thuat_toan": "6 Model V2",
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
        await app.listen({ port: port, host: "0.0.0.0" });
    } catch (err) {
        console.error("Lỗi khởi động:", err);
        process.exit(1);
    }

    console.log("\n" + "=".repeat(60));
    console.log("🚀 SICBO SUNWIN AI SERVER ĐÃ CHẠY!");
    console.log("=".repeat(60));
    console.log(`   ➜ Local:   http://localhost:${port}/`);
    console.log("\n📌 CÁC API ENDPOINTS:");
    console.log(`   ➜ GET /api/sicbo/sunwin   → Dự đoán Tài/Xỉu + 3 Vị`);
    console.log(`   ➜ GET /api/sicbo/history  → Lịch sử`);
    console.log(`   ➜ GET /api/sicbo/stats    → Thống kê`);
    console.log("\n🎯 THÔNG TIN HỆ THỐNG AI:");
    console.log(`   Thuật toán Tài Xỉu: 6 Models V2`);
    console.log(`   Thuật toán 3 Vị: 5 hàm phân tích (giữ nguyên)`);
    console.log("=".repeat(60));
};

start();
