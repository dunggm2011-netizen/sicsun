// ============================================================================
// SUNWIN SICBO PREDICTION ENGINE - FULL API
// Kết hợp: 10 thuật toán + Hack Engine + Dự đoán vị + AI học trọng số
// Deploy: Render (Node.js) | Endpoint: /api/sicbo/sunwin
// ============================================================================

"use strict";

const express = require("express");
const fetch = require("node-fetch");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const API_SOURCE = process.env.API_SOURCE ||
    "https://api.wsktnus8.net/v2/history/getLastResult?gameId=ktrng_3979&size=100&tableId=39791215743193&curPage=1";

const STATE_FILE = path.join(__dirname, "state.json");
const FETCH_INTERVAL = 5000;

// ============================================================================
// GLOBAL STATE
// ============================================================================
let history = [];
let currentSessionId = null;
let lastPrediction = null;
let fetchTimer = null;

const aiState = {
    algoWeights: {},
    algoStats: {},
    patternWeights: {},
    roadStats: {},
    confidenceBuckets: {
        very_high: { total: 0, correct: 0 },
        high: { total: 0, correct: 0 },
        medium: { total: 0, correct: 0 },
        low: { total: 0, correct: 0 },
    },
    total: 0,
    correct: 0,
    consecutiveWrong: 0,
    consecutiveCorrect: 0,
};

// ============================================================================
// UTILS
// ============================================================================
function lastN(arr, n) { return arr.slice(Math.max(0, arr.length - n)); }
function avg(nums) { return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0; }

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

function extractFeatures(history) {
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
    features.meanTotal = avg(totals);
    features.entropy = entropy(tx);
    return features;
}

// ============================================================================
// 10 THUẬT TOÁN
// ============================================================================
function algo1_freqRebalance(h) {
    if (h.length < 15) return null;
    const f = extractFeatures(h);
    const tx = f.tx;
    const t = f.freq['T'] || 0, x = f.freq['X'] || 0;
    if (!tx.length) return null;
    const r10 = tx.slice(-10);
    const r10t = r10.filter(v => v === 'T').length;
    const r10x = r10.filter(v => v === 'X').length;
    if (r10t >= 7) return 'X';
    if (r10x >= 7) return 'T';
    if (t > x + 8 && r10t > r10x + 2) return 'X';
    if (x > t + 8 && r10x > r10t + 2) return 'T';
    if (r10t > r10x + 3) return 'X';
    if (r10x > r10t + 3) return 'T';
    return null;
}

function algo2_markov(h) {
    const tx = extractFeatures(h).tx;
    if (tx.length < 20) return null;
    let best = null, bestConf = 0;
    for (let order = 2; order <= 5; order++) {
        if (tx.length < order + 5) continue;
        const trans = {};
        for (let i = 0; i <= tx.length - order - 1; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            trans[key] = trans[key] || { T: 0, X: 0 };
            trans[key][next]++;
        }
        const key = tx.slice(-order).join('');
        const c = trans[key];
        if (c && c.T + c.X >= 3) {
            const conf = Math.abs(c.T - c.X) / (c.T + c.X);
            if (conf > bestConf && conf > 0.55) {
                bestConf = conf;
                best = c.T > c.X ? 'T' : 'X';
            }
        }
    }
    return best;
}

function algo3_ngram(h) {
    const tx = extractFeatures(h).tx;
    for (let k = 3; k <= 6; k++) {
        if (tx.length < k + 10) continue;
        const last = tx.slice(-k).join('');
        let t = 0, x = 0;
        for (let i = 0; i <= tx.length - k - 1; i++) {
            if (tx.slice(i, i + k).join('') === last) {
                if (tx[i + k] === 'T') t++; else x++;
            }
        }
        const tot = t + x;
        if (tot >= 3 && Math.abs(t - x) / tot >= 0.6) return t > x ? 'T' : 'X';
    }
    return null;
}

function algo4_neoPattern(h) {
    const tx = extractFeatures(h).tx;
    if (tx.length < 40) return null;
    let best = null, bestTotal = 0;
    for (const len of [4, 5, 6]) {
        if (tx.length < len * 2) continue;
        const target = tx.slice(-len);
        let t = 0, x = 0;
        for (let i = 0; i <= tx.length - len - 1; i++) {
            const hist = tx.slice(i, i + len);
            let m = 0;
            for (let j = 0; j < len; j++) if (hist[j] === target[j]) m++;
            if (m / len >= 0.8) {
                if (tx[i + len] === 'T') t++; else x++;
            }
        }
        const tot = t + x;
        if (tot >= 5 && tot > bestTotal && Math.abs(t - x) / tot >= 0.6) {
            bestTotal = tot;
            best = t > x ? 'T' : 'X';
        }
    }
    return best;
}

function algo5_superDeep(h) {
    if (h.length < 80) return null;
    const f = extractFeatures(h);
    const tx = f.tx, totals = f.totals, runs = f.runs;
    const r50 = tx.slice(-50);
    const r50t = r50.filter(v => v === 'T').length;
    const r50x = r50.filter(v => v === 'X').length;
    const avgR = avg(totals.slice(-30));

    let patternPred = null;
    const pat = { 'TTXTT': 'X', 'TXTXT': 'X', 'XXTXX': 'T', 'XTXTX': 'T', 'TTTTX': 'X', 'XXXXT': 'T' };
    const last5 = tx.slice(-5).join('');
    if (pat[last5]) patternPred = pat[last5];

    let runPred = null;
    if (runs.length >= 2) {
        const lr = runs[runs.length - 1];
        if (lr.len >= 4) runPred = lr.val === 'T' ? 'X' : 'T';
    }

    const votes = [];
    if (Math.abs(r50t - r50x) > 10) votes.push(r50t > r50x ? 'T' : 'X');
    if (avgR > 13) votes.push('X');
    else if (avgR < 8.5) votes.push('T');
    if (patternPred) votes.push(patternPred);
    if (runPred) votes.push(runPred);

    if (!votes.length) return null;
    const tV = votes.filter(v => v === 'T').length;
    const xV = votes.filter(v => v === 'X').length;
    if (tV > xV && tV >= 2) return 'T';
    if (xV > tV && xV >= 2) return 'X';
    return null;
}

function algo6_transformer(h) {
    const tx = extractFeatures(h).tx;
    if (tx.length < 60) return null;
    const w = 10;
    const target = tx.slice(-w).join('');
    let tW = 0, xW = 0, totSim = 0;
    for (let i = 0; i <= tx.length - w - 1; i++) {
        const s = similarity(tx.slice(i, i + w).join(''), target);
        if (s > 0.7) {
            const weight = s * (1 / (tx.length - i + 10));
            if (tx[i + w] === 'T') tW += weight; else xW += weight;
            totSim += s;
        }
    }
    if (totSim > 2 && Math.abs(tW - xW) / (tW + xW) > 0.6) return tW > xW ? 'T' : 'X';
    return null;
}

function algo7_superBridge(h) {
    const f = extractFeatures(h);
    const runs = f.runs, tx = f.tx;
    if (runs.length < 4) return null;
    const last5 = runs.slice(-5);
    const r15 = tx.slice(-15);
    const lr = last5[4], prev = last5[3];

    if (lr.len >= 4) return lr.val === 'T' ? 'X' : 'T';
    if (lr.len === 1 && prev.len === 1) {
        let alt = 0;
        for (let i = runs.length - 1; i >= Math.max(0, runs.length - 6); i--) {
            if (runs[i].len === 1) alt++;
        }
        if (alt >= 4) return lr.val === 'T' ? 'X' : 'T';
    }
    const t = r15.filter(v => v === 'T').length;
    const x = r15.filter(v => v === 'X').length;
    if (t > x + 4) return 'X';
    if (x > t + 4) return 'T';
    return null;
}

function algo8_adaptiveMarkov(h) {
    const tx = extractFeatures(h).tx;
    if (tx.length < 25) return null;
    let best = null, bestConf = 0;
    const maxOrder = Math.min(4, Math.floor(tx.length / 10));
    for (let order = 2; order <= maxOrder; order++) {
        if (tx.length < order + 5) continue;
        const trans = {};
        for (let i = 0; i <= tx.length - order - 1; i++) {
            const key = tx.slice(i, i + order).join('');
            const next = tx[i + order];
            trans[key] = trans[key] || { T: 0, X: 0 };
            trans[key][next]++;
        }
        const key = tx.slice(-order).join('');
        const c = trans[key];
        if (c && c.T + c.X >= 2) {
            const conf = Math.abs(c.T - c.X) / (c.T + c.X);
            if (conf > bestConf) {
                bestConf = conf;
                best = c.T > c.X ? 'T' : 'X';
            }
        }
    }
    return bestConf > 0.7 ? best : null;
}

function algo9_neuralPattern(h) {
    const f = extractFeatures(h);
    const tx = f.tx, totals = f.totals;
    if (tx.length < 50) return null;
    const r20 = tx.slice(-20);
    const t20 = r20.filter(v => v === 'T').length;
    const ratio = t20 / 20;
    const avg20 = avg(totals.slice(-20));

    let tS = 0, xS = 0;
    if (ratio > 0.7) xS += 0.35;
    else if (ratio < 0.3) tS += 0.35;
    if (avg20 > 12.5) xS += 0.25;
    else if (avg20 < 8.5) tS += 0.25;

    const pat = { 'TTXTT': { t: 0.2, x: 0.8 }, 'TXTXT': { t: 0.3, x: 0.7 }, 'XXTXX': { t: 0.8, x: 0.2 }, 'XTXTX': { t: 0.7, x: 0.3 }, 'TTTTX': { t: 0.1, x: 0.9 }, 'XXXXT': { t: 0.9, x: 0.1 } };
    const last5 = tx.slice(-5).join('');
    if (pat[last5]) { tS += pat[last5].t * 0.2; xS += pat[last5].x * 0.2; }

    const trend = avg(totals.slice(-10)) - avg(totals.slice(-20, -10));
    if (trend > 2.5) xS += 0.15;
    else if (trend < -2.5) tS += 0.15;

    if (Math.abs(tS - xS) > 0.25) return tS > xS ? 'T' : 'X';
    return null;
}

function algo10_quantum(h) {
    const tx = extractFeatures(h).tx;
    if (tx.length < 40) return null;
    let sT = 0, sX = 0;

    const r30 = tx.slice(-30);
    const t30 = r30.filter(v => v === 'T').length;
    const x30 = r30.filter(v => v === 'X').length;
    if (t30 > x30 + 5) sX += 0.3;
    else if (x30 > t30 + 5) sT += 0.3;

    const last8 = tx.slice(-8).join('');
    let pw = { T: 0, X: 0 };
    for (let i = 0; i <= tx.length - 9; i++) {
        if (tx.slice(i, i + 8).join('') === last8) {
            if (tx[i + 8] === 'T') pw.T++; else pw.X++;
        }
    }
    if (pw.T + pw.X >= 3 && Math.abs(pw.T - pw.X) / (pw.T + pw.X) > 0.6) {
        if (pw.T > pw.X) sT += 0.25; else sX += 0.25;
    }
    const e = entropy(tx.slice(-20));
    if (e > 0.9) {
        if (tx[tx.length - 1] === 'T') sX += 0.2; else sT += 0.2;
    }
    const tot = sT + sX;
    if (tot > 0.5 && Math.abs(sT - sX) / tot > 0.6) return sT > sX ? 'T' : 'X';
    return null;
}

const ALL_ALGS = [
    { id: "freq_rebalance",  fn: algo1_freqRebalance },
    { id: "markov",          fn: algo2_markov },
    { id: "ngram",           fn: algo3_ngram },
    { id: "neo_pattern",     fn: algo4_neoPattern },
    { id: "super_deep",      fn: algo5_superDeep },
    { id: "transformer",     fn: algo6_transformer },
    { id: "super_bridge",    fn: algo7_superBridge },
    { id: "adaptive_markov", fn: algo8_adaptiveMarkov },
    { id: "neural_pattern",  fn: algo9_neuralPattern },
    { id: "quantum",         fn: algo10_quantum },
];

// ============================================================================
// HACK ENGINE - 7 TÍN HIỆU + HỌC CẦU
// ============================================================================
function hackEngine(history) {
    if (history.length < 8) {
        return { prediction: null, scoreT: 0, scoreX: 0, reasons: ["Đang học cầu..."], ready: false };
    }

    const filtered = history.filter(h => h.tx !== 'B');
    const results = filtered.map(h => h.tx).reverse();
    const totals = filtered.map(h => h.total).reverse();

    let scoreT = 0, scoreX = 0;
    const reasons = [];

    // 1. Học pattern 3-4 ký tự
    const patterns = {};
    for (let len = 3; len <= 4; len++) {
        for (let i = 0; i < results.length - len; i++) {
            const key = results.slice(i, i + len).join('');
            const next = results[i + len];
            if (!patterns[key]) patterns[key] = { T: 0, X: 0, total: 0 };
            patterns[key][next]++;
            patterns[key].total++;
        }
    }

    const last4 = results.slice(-4).join('');
    const last3 = results.slice(-3).join('');

    if (patterns[last4] && patterns[last4].total >= 3) {
        const p = patterns[last4];
        const probT = p.T / p.total;
        if (probT >= 0.7) { scoreT += 5; reasons.push(`Cầu4→Tài(${Math.round(probT*100)}%)`); }
        else if (probT <= 0.3) { scoreX += 5; reasons.push(`Cầu4→Xỉu(${Math.round((1-probT)*100)}%)`); }
        else if (probT > 0.5) { scoreT += 2; reasons.push("Cầu4 nhẹ→Tài"); }
        else { scoreX += 2; reasons.push("Cầu4 nhẹ→Xỉu"); }
    } else if (patterns[last3] && patterns[last3].total >= 3) {
        const p = patterns[last3];
        const probT = p.T / p.total;
        if (probT >= 0.7) { scoreT += 4; reasons.push(`Cầu3→Tài(${Math.round(probT*100)}%)`); }
        else if (probT <= 0.3) { scoreX += 4; reasons.push(`Cầu3→Xỉu(${Math.round((1-probT)*100)}%)`); }
        else if (probT > 0.5) { scoreT += 1.5; reasons.push("Cầu3 nhẹ→Tài"); }
        else { scoreX += 1.5; reasons.push("Cầu3 nhẹ→Xỉu"); }
    }

    // 2. Streak
    const last = results[results.length - 1];
    let streak = 1;
    for (let i = results.length - 2; i >= 0; i--) {
        if (results[i] === last) streak++;
        else break;
    }
    if (streak >= 7) {
        if (last === 'T') { scoreX += 5; reasons.push(`Dây Tài ${streak}→Gãy`); }
        else { scoreT += 5; reasons.push(`Dây Xỉu ${streak}→Gãy`); }
    } else if (streak >= 5) {
        if (last === 'T') { scoreX += 3; reasons.push(`Dây Tài ${streak}`); }
        else { scoreT += 3; reasons.push(`Dây Xỉu ${streak}`); }
    } else if (streak >= 3) {
        if (last === 'T') { scoreT += 1; reasons.push(`Theo Tài ${streak}`); }
        else { scoreX += 1; reasons.push(`Theo Xỉu ${streak}`); }
    }

    // 3. Cân bằng 10
    const r10 = results.slice(-10);
    const t10 = r10.filter(r => r === 'T').length;
    const x10 = r10.length - t10;
    if (t10 >= 8) { scoreX += 4; reasons.push(`10p lệch Tài(${t10})`); }
    if (x10 >= 8) { scoreT += 4; reasons.push(`10p lệch Xỉu(${x10})`); }
    if (t10 >= 7 && t10 < 8) { scoreX += 2; reasons.push(`10p hơi Tài`); }
    if (x10 >= 7 && x10 < 8) { scoreT += 2; reasons.push(`10p hơi Xỉu`); }

    // 4. Tổng điểm gần nhất
    const lastTotal = totals[totals.length - 1];
    if (lastTotal >= 16) { scoreX += 5; reasons.push(`Tổng rất cao ${lastTotal}`); }
    else if (lastTotal <= 5) { scoreT += 5; reasons.push(`Tổng rất thấp ${lastTotal}`); }
    else if (lastTotal >= 14) { scoreX += 3; reasons.push(`Tổng cao ${lastTotal}`); }
    else if (lastTotal <= 7) { scoreT += 3; reasons.push(`Tổng thấp ${lastTotal}`); }
    else if (lastTotal >= 12) { scoreX += 1; reasons.push(`Tổng hơi cao`); }
    else if (lastTotal <= 9) { scoreT += 1; reasons.push(`Tổng hơi thấp`); }

    // 5. MA
    const avg3 = avg(totals.slice(-3));
    const avg7 = avg(totals.slice(-7));
    if (avg3 > avg7 + 2) { scoreX += 2; reasons.push("MA↑mạnh"); }
    else if (avg3 < avg7 - 2) { scoreT += 2; reasons.push("MA↓mạnh"); }

    // 6. Zigzag
    let zigzag = true;
    const r6 = results.slice(-6);
    for (let i = 1; i < r6.length; i++) {
        if (r6[i] === r6[i - 1]) { zigzag = false; break; }
    }
    if (zigzag && r6.length >= 6) {
        if (r6[r6.length - 1] === 'T') { scoreX += 2; reasons.push("Zigzag→Xỉu"); }
        else { scoreT += 2; reasons.push("Zigzag→Tài"); }
    }

    // 7. Auto-correct
    if (aiState.consecutiveWrong >= 3) {
        if (last === 'T') { scoreX += 6; reasons.push("⚡FIX→Xỉu"); }
        else { scoreT += 6; reasons.push("⚡FIX→Tài"); }
    } else if (aiState.consecutiveWrong >= 2) {
        if (last === 'T') { scoreX += 3; reasons.push("⚡FIX nhẹ→Xỉu"); }
        else { scoreT += 3; reasons.push("⚡FIX nhẹ→Tài"); }
    }

    // Quyết định
    let prediction;
    if (scoreT > scoreX) prediction = 'T';
    else if (scoreX > scoreT) prediction = 'X';
    else prediction = last === 'T' ? 'X' : 'T';

    return { prediction, scoreT, scoreX, reasons, ready: true };
}

// ============================================================================
// DỰ ĐOÁN VỊ
// ============================================================================
function predictScores(history, txConstraint) {
    const xiuScores = [4, 5, 6, 7, 8, 9, 10];
    const taiScores = [11, 12, 13, 14, 15, 16, 17];
    const range = txConstraint === 'T' ? taiScores : xiuScores;

    const filtered = history.filter(h => h.tx !== 'B');
    if (filtered.length < 15) {
        return {
            top3: [range[0], range[Math.floor(range.length / 2)], range[range.length - 1]],
            top5: range.slice(0, 5),
            probabilities: {},
        };
    }

    const scores = {};
    range.forEach(s => scores[s] = 0.1);

    // Frequency có decay
    const lastN = filtered.slice(-60);
    lastN.forEach((h, i) => {
        if (h.tx === txConstraint && range.includes(h.total)) {
            const age = lastN.length - i - 1;
            scores[h.total] += Math.exp(-age / 20);
        }
    });

    // Gap
    range.forEach(s => {
        let lastSeen = -1;
        for (let i = filtered.length - 1; i >= 0; i--) {
            if (filtered[i].tx === txConstraint && filtered[i].total === s) { lastSeen = i; break; }
        }
        if (lastSeen === -1) scores[s] += 0.7;
        else {
            const gap = filtered.length - lastSeen;
            scores[s] += Math.min(1, gap / 15);
        }
    });

    // Cluster
    filtered.filter(h => h.tx === txConstraint).slice(-30).forEach(h => {
        if (scores[h.total] !== undefined) scores[h.total] += 0.3;
    });

    const tot = Object.values(scores).reduce((a, b) => a + b, 0);
    range.forEach(s => scores[s] /= tot);

    // Chọn top 3 đa dạng
    const ranked = Object.entries(scores)
        .map(([s, p]) => ({ score: parseInt(s), prob: p }))
        .sort((a, b) => b.prob - a.prob);

    const mid = Math.floor((range[0] + range[range.length - 1]) / 2);
    const top3 = [ranked[0].score];
    for (const r of ranked) {
        if (r.score < mid && !top3.includes(r.score)) { top3.push(r.score); break; }
    }
    for (const r of ranked) {
        if (r.score > mid && !top3.includes(r.score)) { top3.push(r.score); break; }
    }
    for (const r of ranked) {
        if (top3.length >= 3) break;
        if (!top3.includes(r.score)) top3.push(r.score);
    }

    return {
        top3: top3.sort((a, b) => a - b),
        top5: ranked.slice(0, 5).map(x => x.score).sort((a, b) => a - b),
        probabilities: scores,
    };
}

// ============================================================================
// AI WEIGHT LEARNING
// ============================================================================
function initAi() {
    for (const alg of ALL_ALGS) {
        if (!aiState.algoWeights[alg.id]) {
            aiState.algoWeights[alg.id] = 1.0;
            aiState.algoStats[alg.id] = { total: 0, correct: 0 };
        }
    }
}

function updateAiWeights(actualTx, predictedTx) {
    if (actualTx === 'B' || !predictedTx) return;

    aiState.total++;
    const isCorrect = predictedTx === actualTx;
    if (isCorrect) {
        aiState.correct++;
        aiState.consecutiveCorrect++;
        aiState.consecutiveWrong = 0;
    } else {
        aiState.consecutiveWrong++;
        aiState.consecutiveCorrect = 0;
    }

    // Cập nhật weight từng thuật toán
    for (const alg of ALL_ALGS) {
        const pred = alg.fn(history.slice(0, -1));
        if (!pred) continue;
        const s = aiState.algoStats[alg.id];
        s.total++;
        if (pred === actualTx) s.correct++;

        const acc = s.total > 0 ? s.correct / s.total : 0.5;
        const target = Math.pow(acc, 2);
        aiState.algoWeights[alg.id] = Math.max(0.1, aiState.algoWeights[alg.id] * 0.85 + target * 0.15);
    }

    // Confidence bucket
    if (lastPrediction && lastPrediction.confidence) {
        const c = lastPrediction.confidence / 100;
        const bucket = c > 0.75 ? 'very_high' : c > 0.65 ? 'high' : c > 0.55 ? 'medium' : 'low';
        aiState.confidenceBuckets[bucket].total++;
        if (isCorrect) aiState.confidenceBuckets[bucket].correct++;
    }

    saveState();
}

// ============================================================================
// ENSEMBLE PREDICTION
// ============================================================================
function ensemblePredict(history) {
    const votes = { T: 0, X: 0, none: 0 };
    const details = {};

    for (const alg of ALL_ALGS) {
        const pred = alg.fn(history);
        const weight = aiState.algoWeights[alg.id] || 1;
        details[alg.id] = { pred, weight: weight.toFixed(3) };
        if (pred === 'T') votes.T += weight;
        else if (pred === 'X') votes.X += weight;
        else votes.none++;
    }

    // Hack engine vote
    const hack = hackEngine(history);
    if (hack.prediction) {
        const hackScore = Math.max(hack.scoreT, hack.scoreX);
        const w = Math.min(3, hackScore / 3);
        if (hack.prediction === 'T') votes.T += w;
        else votes.X += w;
    }

    const total = votes.T + votes.X;
    if (total === 0) return { prediction: 'T', confidence: 0.5, votes, details, hack };

    const prediction = votes.T > votes.X ? 'T' : 'X';
    const confidence = Math.max(votes.T, votes.X) / total;

    return { prediction, confidence, votes, details, hack };
}

// ============================================================================
// PARSE INPUT
// ============================================================================
function parseResult(data) {
    if (!data || !data.data || !Array.isArray(data.data.resultList)) return [];

    const sorted = data.data.resultList.sort((a, b) => {
        return parseInt(b.gameNum.slice(1)) - parseInt(a.gameNum.slice(1));
    });

    return sorted.map(item => {
        const total = item.score;
        let tx, result;
        if (total >= 4 && total <= 10) { tx = 'X'; result = 'XIU'; }
        else if (total >= 11 && total <= 17) { tx = 'T'; result = 'TAI'; }
        else if (total === 3 || total === 18) { tx = 'B'; result = 'BAO'; }
        else { tx = 'N'; result = 'UNKNOWN'; }

        const dice = Array.isArray(item.facesList)
            ? item.facesList
            : (typeof item.keyR === 'string' ? item.keyR.split('-').map(Number) : [0, 0, 0]);

        return {
            session: parseInt(item.gameNum.slice(1)),
            dice, total, result, tx,
        };
    }).sort((a, b) => a.session - b.session);
}

// ============================================================================
// FETCH LOOP
// ============================================================================
async function fetchData() {
    try {
        const res = await fetch(API_SOURCE, {
            headers: { "User-Agent": "Mozilla/5.0", "Accept": "application/json" },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const newHistory = parseResult(data);
        if (!newHistory.length) return;

        const last = newHistory[newHistory.length - 1];

        if (!currentSessionId) {
            history = newHistory;
            currentSessionId = last.session;
            initAi();
            // Warm-up AI
            for (let i = 20; i < newHistory.length; i++) {
                const prefix = newHistory.slice(0, i);
                const actual = newHistory[i].tx;
                // Cập nhật stats cho từng thuật toán
                for (const alg of ALL_ALGS) {
                    const pred = alg.fn(prefix);
                    if (!pred) continue;
                    const s = aiState.algoStats[alg.id];
                    s.total++;
                    if (pred === actual) s.correct++;
                }
            }
            // Init weights
            for (const alg of ALL_ALGS) {
                const s = aiState.algoStats[alg.id];
                const acc = s.total > 0 ? s.correct / s.total : 0.5;
                aiState.algoWeights[alg.id] = Math.max(0.1, Math.pow(acc, 2));
            }
            console.log(`[Init] Loaded ${newHistory.length} sessions, session ID: ${currentSessionId}`);
        } else if (last.session > currentSessionId) {
            const newRecords = newHistory.filter(r => r.session > currentSessionId);
            for (const rec of newRecords) {
                // Đánh giá dự đoán trước
                if (lastPrediction && lastPrediction.session === rec.session) {
                    updateAiWeights(rec.tx, lastPrediction.prediction);
                }
                history.push(rec);
            }
            if (history.length > 500) history = history.slice(-500);
            currentSessionId = last.session;
            console.log(`[Fetch] +${newRecords.length} sessions, current: ${currentSessionId}`);
        }

        // Tạo dự đoán mới
        makePrediction();
        saveState();

    } catch (e) {
        console.error(`[Fetch] Error: ${e.message}`);
    }
}

function makePrediction() {
    if (history.length < 10) return;

    const result = ensemblePredict(history);
    const posResult = predictScores(history, result.prediction);

    lastPrediction = {
        session: currentSessionId + 1,
        prediction: result.prediction,
        confidence: Math.round(result.confidence * 100),
        top3: posResult.top3,
        top5: posResult.top5,
        votes: result.votes,
        details: result.details,
        hack: result.hack,
        timestamp: new Date().toISOString(),
    };
}

// ============================================================================
// STATE PERSISTENCE
// ============================================================================
function saveState() {
    try {
        const data = {
            aiState: {
                algoWeights: aiState.algoWeights,
                algoStats: aiState.algoStats,
                confidenceBuckets: aiState.confidenceBuckets,
                total: aiState.total,
                correct: aiState.correct,
            },
            currentSessionId,
        };
        fs.writeFileSync(STATE_FILE, JSON.stringify(data, null, 2));
    } catch (e) {
        console.error(`[Save] Error: ${e.message}`);
    }
}

function loadState() {
    try {
        if (fs.existsSync(STATE_FILE)) {
            const data = JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
            if (data.aiState) {
                Object.assign(aiState, data.aiState);
            }
            if (data.currentSessionId) currentSessionId = data.currentSessionId;
            console.log(`[State] Loaded, accuracy: ${getAccuracy()}%`);
        }
    } catch (e) {
        console.error(`[Load] Error: ${e.message}`);
    }
}

function getAccuracy() {
    if (aiState.total === 0) return 0;
    return (aiState.correct / aiState.total * 100).toFixed(1);
}

// ============================================================================
// API ENDPOINTS
// ============================================================================
app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    next();
});

app.get("/api/sicbo/sunwin", (req, res) => {
    const last = history[history.length - 1];
    const pred = lastPrediction;

    if (!last || !pred) {
        return res.json({
            status: "waiting",
            msg: "Đang tải dữ liệu",
            phien_hien_tai: currentSessionId,
        });
    }

    const topAlgos = Object.entries(aiState.algoWeights)
        .map(([id, w]) => {
            const s = aiState.algoStats[id] || { total: 0, correct: 0 };
            const acc = s.total > 0 ? (s.correct / s.total * 100).toFixed(1) : "N/A";
            return { id, weight: w.toFixed(3), accuracy: acc + "%", samples: s.total };
        })
        .sort((a, b) => parseFloat(b.weight) - parseFloat(a.weight))
        .slice(0, 5);

    res.json({
        id: "Sicbo Sunwin AI",
        phien_truoc: last.session,
        xuc_xac1: last.dice[0],
        xuc_xac2: last.dice[1],
        xuc_xac3: last.dice[2],
        tong: last.total,
        ket_qua: last.result.toLowerCase(),
        phien_hien_tai: currentSessionId + 1,

        du_doan: pred.prediction === 'T' ? 'tài' : 'xỉu',
        do_tin_cay: pred.confidence + '%',

        du_doan_vi: pred.top3.join('-'),
        du_doan_top5: pred.top5.join('-'),

        votes: pred.votes,
        top_algos: topAlgos,

        accuracy: getAccuracy() + "%",
        tong_du_doan: aiState.total,
    });
});

app.get("/api/sicsun/history", (req, res) => {
    const reversed = [...history].sort((a, b) => b.session - a.session);
    const result = [];
    for (const h of reversed) {
        const item = {
            session: h.session,
            dice: h.dice,
            total: h.total,
            result: h.result.toLowerCase(),
            tx_label: h.tx.toLowerCase(),
        };
        result.push(item);
        result.push(item);
    }
    res.json(result);
});

app.get("/api/sicbo/stats", (req, res) => {
    const algos = {};
    for (const alg of ALL_ALGS) {
        const s = aiState.algoStats[alg.id] || { total: 0, correct: 0 };
        algos[alg.id] = {
            weight: (aiState.algoWeights[alg.id] || 1).toFixed(3),
            total: s.total,
            correct: s.correct,
            accuracy: s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + "%" : "N/A",
        };
    }

    const buckets = {};
    for (const [k, v] of Object.entries(aiState.confidenceBuckets)) {
        buckets[k] = {
            total: v.total,
            correct: v.correct,
            accuracy: v.total > 0 ? (v.correct / v.total * 100).toFixed(1) + "%" : "N/A",
        };
    }

    res.json({
        tong_phien: history.length,
        phien_hien_tai: currentSessionId,
        do_chinh_xac: getAccuracy() + "%",
        tong_du_doan: aiState.total,
        so_thuat_toan: ALL_ALGS.length,
        thong_ke_thuat_toan: algos,
        thong_ke_tin_cay: buckets,
        consecutive: {
            wrong: aiState.consecutiveWrong,
            correct: aiState.consecutiveCorrect,
        },
    });
});

app.get("/api/sicbo/predict", (req, res) => {
    if (!lastPrediction) return res.json({ status: "waiting" });
    res.json(lastPrediction);
});

app.get("/", (req, res) => {
    res.json({
        status: "ok",
        service: "Sicbo Sunwin AI",
        version: "1.0",
        endpoints: {
            "/api/sicbo/sunwin": "Dự đoán Tài/Xỉu + vị",
            "/api/sicsun/history": "Lịch sử",
            "/api/sicbo/stats": "Thống kê",
            "/api/sicbo/predict": "Chi tiết dự đoán",
        },
        thong_tin: {
            so_thuat_toan: ALL_ALGS.length,
            phien_hien_tai: currentSessionId,
            tong_phien: history.length,
            accuracy: getAccuracy() + "%",
        },
    });
});

// ============================================================================
// START
// ============================================================================
loadState();
initAi();
fetchData();
setInterval(fetchData, FETCH_INTERVAL);

app.listen(PORT, "0.0.0.0", () => {
    console.log("=".repeat(60));
    console.log("🚀 SICBO SUNWIN AI API");
    console.log("=".repeat(60));
    console.log(`   Port: ${PORT}`);
    console.log(`   API: http://localhost:${PORT}/api/sicbo/sunwin`);
    console.log(`   Source: ${API_SOURCE}`);
    console.log("=".repeat(60));
});
