/**
 * Statistics transformation and calculation utilities for benchmark analysis
 */

const EXPECTED_FLAGS = [
  "length_ok",
  "same_meaning",
  "same_lang",
  "no_missing_info",
  "citation_preserved",
  "citation_content_ok",
];

/**
 * Format a number with specified decimal places
 */
export const formatMetric = (value, decimals = 1) => {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return value.toFixed(decimals);
};

/**
 * Format percentage
 */
export const formatPct = (value, decimals = 1) => {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return `${value.toFixed(decimals)}%`;
};

/**
 * Create detailed model comparison table data from folder stats
 * Returns array of rows with all 33 columns
 */
export const createModelComparisonTable = (stats, folder) => {
  if (!stats || !stats[folder]) return [];

  const rows = [];
  const folderStats = stats[folder];

  Object.entries(folderStats).forEach(([model, modes]) => {
    ["doc", "para"].forEach((mode) => {
      if (!modes[mode]) return;

      const s = modes[mode];
      const baselineGz = s.baseline?.gptzero || 0;
      const baselineSp = s.baseline?.sapling || 0;
      const afterGz = s.after?.gptzero || 0;
      const afterSp = s.after?.sapling || 0;

      const row = {
        // Basic info (1-4)
        model,
        mode: mode.charAt(0).toUpperCase() + mode.slice(1),
        drafts: s.draft_count || 0,
        paragraphs: s.total_content_paragraphs || 0,

        // AI Detection (5-12)
        baselineGz,
        afterGz,
        deltaGz: s.deltas?.gptzero || 0,
        zeroshotGz: s.zero_shot_success?.gptzero || 0,
        baselineSp,
        afterSp,
        deltaSp: s.deltas?.sapling || 0,
        zeroshotSp: s.zero_shot_success?.sapling || 0,

        // Length deviations (13-17)
        avgDraftDeltaPct: s.draft_length_deviation_avg || 0,
        avgParaDeltaPct: s.para_length_deviation_avg || 0,
        lenWithin10Pct: s.length_within_10_pct || 0,
        lenWithin15Pct: s.length_within_15_pct || 0,
        lenWithin20Pct: s.length_within_20_pct || 0,

        // Word count metrics (18-22)
        avgWcDelta: s.deltas?.wordcount || 0,
        within10Words: s.wc_diff?.within10 || 0,
        within20Words: s.wc_diff?.within20 || 0,
        pctLonger: s.wc_diff?.pct_longer || 0,
        pctShorter: s.wc_diff?.pct_shorter || 0,

        // Quality metrics (23-27)
        qualityPct: s.quality
          ? Object.values(s.quality).reduce((a, b) => a + b, 0) / Object.values(s.quality).length
          : 0,
        grammarLv: s.grammar_score,
        draftsWithParaSplitPct: s.draft_with_para_mismatch_pct || 0,
        draftsWithDocMismatchPct: s.draft_with_doc_mismatch_pct || 0,
        mismatchedParagraphsPct: s.mismatched_paragraphs_pct || 0,

        // NEW numeric quality levels (26-27)
        sameMeaningLv: s.same_meaning_level_avg,
        missingInfoLv: s.missing_info_level_avg,

        // Boolean quality flags (28-31)
        lengthOk: s.quality?.length_ok || 0,
        sameMeaning: s.quality?.same_meaning || 0,
        sameLang: s.quality?.same_lang || 0,
        noMissingInfo: s.quality?.no_missing_info || 0,
        citationPreserved: s.quality?.citation_preserved || 0,
        citationContentOk: s.quality?.citation_content_ok || 0,

        // Citation preservation metrics (32-33)
        citationPreservedPct: s.citation_preservation_rate_avg || 100,
        citationExactPct: s.citation_exact_match_rate_avg || 100,

        // Style Adherence (34)
        styleAdherence: s.style_adherence?.overall ?? null,
        styleAdherenceCount: s.style_adherence?.count ?? 0,
      };

      rows.push(row);
    });
  });

  return rows;
};

/**
 * Get color class for delta values (lower is better)
 */
export const getDeltaColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value < -0.1) return "text-green-600 font-semibold";
  if (value < 0) return "text-green-500";
  if (value > 0.1) return "text-red-600 font-semibold";
  if (value > 0) return "text-red-500";
  return "";
};

/**
 * Get color class for zero-shot percentage (higher is better)
 */
export const getZeroShotColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 80) return "text-green-600 font-semibold";
  if (value >= 50) return "text-orange-500";
  return "text-red-600";
};

/**
 * Get color class for quality percentage (higher is better)
 */
export const getQualityColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 90) return "text-green-600";
  if (value >= 70) return "text-orange-500";
  return "text-red-600";
};

/**
 * Get color class for grammar level (0-10 scale, higher is better)
 */
export const getGrammarColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 8) return "text-green-600 font-semibold";
  if (value >= 6) return "text-orange-500";
  if (value < 4) return "text-red-600 font-semibold";
  return "";
};

/**
 * Get color class for meaning level (0-10 scale, higher is better)
 */
export const getMeaningLevelColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 8) return "text-green-600 font-semibold";
  if (value >= 6) return "text-orange-500";
  if (value < 4) return "text-red-600 font-semibold";
  return "";
};

/**
 * Get color class for missing info level (0-10 scale, lower is better)
 */
export const getMissingInfoColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value <= 2) return "text-green-600 font-semibold";
  if (value <= 4) return "text-orange-500";
  if (value > 6) return "text-red-600 font-semibold";
  return "";
};

/**
 * Get color class for citation metrics (higher is better)
 */
export const getCitationColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 95) return "text-green-600 font-semibold";
  if (value >= 80) return "text-orange-500";
  if (value < 60) return "text-red-600 font-semibold";
  return "";
};

/**
 * Get color class for length deviation percentage (higher is better - within range)
 */
export const getLengthDeviationColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 80) return "text-green-600 font-semibold";
  if (value >= 60) return "text-orange-500";
  return "text-red-600";
};

/**
 * Get color class for style adherence score (0-10 scale, higher is better)
 */
export const getStyleAdherenceColor = (value) => {
  if (value === null || value === undefined || isNaN(value)) return "";
  if (value >= 8) return "text-green-600 font-semibold";
  if (value >= 6) return "text-orange-500";
  if (value < 4) return "text-red-600 font-semibold";
  return "";
};

/**
 * Compute model performance summary across folders
 */
export const computeModelPerformance = (stats, restrictFolders = null) => {
  const agg = {};

  Object.entries(stats).forEach(([folder, models]) => {
    if (restrictFolders && !restrictFolders.includes(folder)) return;

    Object.entries(models).forEach(([model, modes]) => {
      Object.entries(modes).forEach(([mode, s]) => {
        const key = `${model}-${mode}`;
        if (!agg[key]) {
          agg[key] = {
            model,
            mode,
            gzDeltas: [],
            spDeltas: [],
            quality: [],
            grammarScores: [],
            drafts: 0,
            zsGzHits: 0,
            zsSpHits: 0,
            folders: new Set(),
            sourceRuns: new Set(),
          };
        }

        const bucket = agg[key];
        bucket.gzDeltas.push(s.deltas?.gptzero || 0);
        if (s.deltas?.sapling != null && !isNaN(s.deltas.sapling)) {
          bucket.spDeltas.push(s.deltas.sapling);
        }
        bucket.quality.push(
          s.quality
            ? Object.values(s.quality).reduce((a, b) => a + b, 0) / Object.values(s.quality).length
            : 0
        );

        if (s.grammar_score != null) {
          bucket.grammarScores.push(s.grammar_score);
        }

        bucket.drafts += s.draft_count || 0;
        bucket.zsGzHits += s.zs_hits?.gptzero || 0;
        bucket.zsSpHits += s.zs_hits?.sapling || 0;
        bucket.folders.add(folder);

        if (s.source_runs) {
          s.source_runs.forEach((run) => bucket.sourceRuns.add(run));
        }
      });
    });
  });

  const rows = Object.values(agg)
    .filter((m) => m.drafts > 0)
    .map((m) => ({
      model: m.model,
      mode: m.mode.charAt(0).toUpperCase() + m.mode.slice(1),
      totalDrafts: m.drafts,
      avgDeltaGz: m.gzDeltas.length > 0 ? m.gzDeltas.reduce((a, b) => a + b, 0) / m.gzDeltas.length : 0,
      avgDeltaSp: m.spDeltas.length > 0 ? m.spDeltas.reduce((a, b) => a + b, 0) / m.spDeltas.length : 0,
      zeroshotGz: (m.zsGzHits / m.drafts) * 100,
      zeroshotSp: (m.zsSpHits / m.drafts) * 100,
      avgQuality: m.quality.length > 0 ? m.quality.reduce((a, b) => a + b, 0) / m.quality.length : 0,
      avgGrammar:
        m.grammarScores.length > 0
          ? m.grammarScores.reduce((a, b) => a + b, 0) / m.grammarScores.length
          : null,
      folders: m.folders.size,
      sourceRuns: m.sourceRuns.size,
    }));

  return rows;
};

/**
 * Describe array with statistical measures
 */
const describe = (arr) => {
  if (!arr || arr.length === 0) {
    return { min: 0, p25: 0, median: 0, mean: 0, p75: 0, max: 0 };
  }

  const clean = arr.filter((v) => v != null && !isNaN(v));
  if (clean.length === 0) {
    return { min: 0, p25: 0, median: 0, mean: 0, p75: 0, max: 0 };
  }

  const sorted = [...clean].sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;

  return {
    min: sorted[0],
    p25: sorted[Math.floor(sorted.length * 0.25)],
    median: sorted[Math.floor(sorted.length * 0.5)],
    mean,
    p75: sorted[Math.floor(sorted.length * 0.75)],
    max: sorted[sorted.length - 1],
  };
};

/**
 * Build extended statistics with descriptive measures
 */
export const buildExtendedStats = (stats) => {
  const rows = [];

  Object.entries(stats).forEach(([folder, models]) => {
    Object.entries(models).forEach(([model, modes]) => {
      Object.entries(modes).forEach(([mode, s]) => {
        const series = s.series || {};
        if (Object.keys(series).length === 0) return;

        const grammarSeries = series.grammar || [];

        const row = {
          folder,
          model,
          mode: mode.charAt(0).toUpperCase() + mode.slice(1),
          gptZero: describe(series.after_gz || []),
          sapling: describe(series.after_sp || []),
          wcDelta: describe(series.wc || []),
          quality: describe(series.quality || []),
          grammarLv: describe(grammarSeries),
        };

        rows.push(row);
      });
    });
  });

  return rows;
};
