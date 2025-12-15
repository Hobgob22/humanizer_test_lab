import { useState, useEffect } from "react";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/Alert";
import { formatNumber, formatPercent, downloadJSON } from "../lib/utils";
import {
  createModelComparisonTable,
  computeModelPerformance,
  buildExtendedStats,
  formatMetric,
  formatPct,
  getDeltaColor,
  getZeroShotColor,
  getQualityColor,
  getGrammarColor,
  getMeaningLevelColor,
  getMissingInfoColor,
  getCitationColor,
  getLengthDeviationColor,
  getStyleAdherenceColor,
} from "../lib/statistics";
import { Loader2, Download, BarChart3, Info, ChevronDown, ChevronUp, Trash2 } from "lucide-react";

export function BenchmarkAnalysis() {
  const RUNS_COLLAPSED_LIMIT = 12;
  const [runs, setRuns] = useState([]);
  const [selectedRuns, setSelectedRuns] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(false);
  const [error, setError] = useState(null);
  const [mergeRuns, setMergeRuns] = useState(false);
  const [activeTab, setActiveTab] = useState("folder");
  const [deleting, setDeleting] = useState(false);
  const [showAllRuns, setShowAllRuns] = useState(false);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      setError(null);
      const data = await api.listRuns();
      setRuns(data?.runs || []);
    } catch (err) {
      setError(err.message || "Failed to load runs");
      setRuns([]);
    } finally {
      setLoading(false);
    }
  };

  const handleRunSelection = (runName) => {
    const newSelection = selectedRuns.includes(runName)
      ? selectedRuns.filter((r) => r !== runName)
      : [...selectedRuns, runName];
    setSelectedRuns(newSelection);
  };

  const computeStatistics = async () => {
    if (selectedRuns.length === 0) {
      alert("Please select at least one run");
      return;
    }

    setLoadingStats(true);
    try {
      const response = await api.computeStatistics(selectedRuns, mergeRuns);

      // Poll for completion if async
      if (response.task_id) {
        const pollInterval = setInterval(async () => {
          try {
            const status = await api.getStatisticsStatus(response.task_id);
            if (status.status === "completed") {
              clearInterval(pollInterval);
              // Keep the nested structure for display
              setStatistics(status.result);
              setLoadingStats(false);
            } else if (status.status === "failed") {
              clearInterval(pollInterval);
              alert(`Statistics computation failed: ${status.error}`);
              setLoadingStats(false);
            }
          } catch (err) {
            clearInterval(pollInterval);
            alert(`Failed to get statistics status: ${err.message}`);
            setLoadingStats(false);
          }
        }, 1000);
      } else {
        setStatistics(response.result);
        setLoadingStats(false);
      }
    } catch (err) {
      alert(`Failed to compute statistics: ${err.message}`);
      setLoadingStats(false);
    }
  };

  const handleDownloadData = () => {
    const exportData = {
      runs: selectedRuns,
      merge: mergeRuns,
      statistics,
      timestamp: Date.now(),
    };
    downloadJSON(exportData, `benchmark_export_${Date.now()}.json`);
  };

  const handleDeleteRuns = async () => {
    if (selectedRuns.length === 0) {
      alert("Please select at least one run to delete");
      return;
    }

    const confirmMessage = selectedRuns.length === 1
      ? `Are you sure you want to delete the run "${selectedRuns[0]}"?`
      : `Are you sure you want to delete ${selectedRuns.length} runs?\n\n${selectedRuns.join(", ")}`;

    if (!window.confirm(confirmMessage)) {
      return;
    }

    setDeleting(true);
    setError(null);

    try {
      let successCount = 0;
      let failCount = 0;
      const errors = [];

      for (const runName of selectedRuns) {
        try {
          await api.deleteRun(runName);
          successCount++;
        } catch (err) {
          failCount++;
          errors.push(`${runName}: ${err.message}`);
        }
      }

      if (successCount > 0) {
        // Refresh the runs list
        await loadRuns();
        // Clear selection and statistics
        setSelectedRuns([]);
        setStatistics(null);
      }

      if (failCount === 0) {
        alert(`Successfully deleted ${successCount} run${successCount > 1 ? "s" : ""}`);
      } else {
        alert(
          `Deleted ${successCount} run${successCount > 1 ? "s" : ""}, failed to delete ${failCount}:\n\n${errors.join("\n")}`
        );
      }
    } catch (err) {
      setError(`Failed to delete runs: ${err.message}`);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  const hasHiddenRuns = runs.length > RUNS_COLLAPSED_LIMIT;
  const visibleRuns = showAllRuns ? runs : runs.slice(0, RUNS_COLLAPSED_LIMIT);
  const hiddenRunCount = Math.max(runs.length - RUNS_COLLAPSED_LIMIT, 0);

  const tabs = [
    { id: "folder", label: "📊 By Folder & Model" },
    { id: "performance", label: "📈 Model Performance" },
    { id: "extended", label: "📐 Extended Stats" },
    { id: "folderSummary", label: "📁 Folder Summary" },
    { id: "distributions", label: "📊 Distributions" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Benchmark Analysis</h1>
          <p className="text-muted-foreground mt-1">Analyze and compare humanization benchmarks</p>
        </div>
        <Button onClick={loadRuns} variant="outline" size="sm">
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Run Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Runs</CardTitle>
          <CardDescription>Choose one or more runs to analyze ({selectedRuns.length} selected)</CardDescription>
        </CardHeader>
        <CardContent>
          {runs.length === 0 ? (
            <p className="text-center text-muted-foreground py-8">No runs available</p>
          ) : (
            <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {visibleRuns.map((run) => (
                <div key={run.name} className="flex items-center gap-2 p-2 border rounded">
                  <input
                    type="checkbox"
                    id={run.name}
                    checked={selectedRuns.includes(run.name)}
                    onChange={() => handleRunSelection(run.name)}
                  />
                  <label htmlFor={run.name} className="text-sm cursor-pointer flex-1">
                    <div className="font-medium">{run.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(run.timestamp * 1000).toLocaleDateString()}
                    </div>
                  </label>
                </div>
              ))}
            </div>

              <div className="flex items-center justify-between mt-4">
                <p className="text-sm text-muted-foreground">
                  {showAllRuns
                    ? `Showing all ${runs.length} runs`
                    : `Showing latest ${visibleRuns.length} of ${runs.length} runs`}
                </p>
                {hasHiddenRuns && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAllRuns((prev) => !prev)}
                  >
                    {showAllRuns ? (
                      <>
                        <ChevronUp className="h-4 w-4 mr-1" />
                        Show fewer
                      </>
                    ) : (
                      <>
                        <ChevronDown className="h-4 w-4 mr-1" />
                        Show all runs ({hiddenRunCount} more)
                      </>
                    )}
                  </Button>
                )}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Analysis Options */}
      {selectedRuns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Options</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="mergeRuns"
                checked={mergeRuns}
                onChange={(e) => setMergeRuns(e.target.checked)}
              />
              <label htmlFor="mergeRuns" className="text-sm font-medium cursor-pointer">
                Merge runs into combined analysis
              </label>
            </div>

            <div className="flex gap-3 flex-wrap">
              <Button onClick={computeStatistics} disabled={loadingStats}>
                {loadingStats ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Computing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Compute Statistics
                  </>
                )}
              </Button>
              {statistics && (
                <Button variant="outline" onClick={handleDownloadData}>
                  <Download className="h-4 w-4 mr-2" />
                  Download Data
                </Button>
              )}
              <Button
                variant="destructive"
                onClick={handleDeleteRuns}
                disabled={deleting || selectedRuns.length === 0}
              >
                {deleting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete Selected ({selectedRuns.length})
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics Results */}
      {statistics && (
        <Card>
          <CardHeader>
            <CardTitle>Statistical Analysis</CardTitle>
            <CardDescription>
              Analysis of {selectedRuns.length} run{selectedRuns.length > 1 ? "s" : ""}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Tab Navigation */}
            <div className="flex gap-2 border-b mb-6 overflow-x-auto">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? "border-primary text-primary"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            {activeTab === "folder" && <FolderModelView stats={statistics} />}
            {activeTab === "performance" && <ModelPerformanceView stats={statistics} />}
            {activeTab === "extended" && <ExtendedStatsView stats={statistics} />}
            {activeTab === "folderSummary" && <FolderSummaryView stats={statistics} />}
            {activeTab === "distributions" && <DistributionsView stats={statistics} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ====== Tab Components ======

function FolderModelView({ stats }) {
  const [expandedFolders, setExpandedFolders] = useState(["ai_texts"]);

  const toggleFolder = (folder) => {
    setExpandedFolders((prev) =>
      prev.includes(folder) ? prev.filter((f) => f !== folder) : [...prev, folder]
    );
  };

  const folderOrder = ["ai_texts", "human_texts", "ai_paras", "human_paras"];
  const availableFolders = folderOrder.filter((f) => stats[f]);
  const otherFolders = Object.keys(stats).filter((f) => !folderOrder.includes(f));
  const allFolders = [...availableFolders, ...otherFolders];

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="font-medium mb-2 flex items-center gap-2">
          <Info className="h-4 w-4" />
          Understanding the metrics
        </h3>
        <div className="text-sm space-y-1 text-muted-foreground">
          <p>
            <strong>Δ GZ / Δ SP</strong> – change in AI-detection score (negative = better)
          </p>
          <p>
            <strong>Zero-shot</strong> – % drafts ≤ 10% on detector
          </p>
          <p>
            <strong>Quality %</strong> – average of all quality checks
          </p>
          <p>
            <strong>Grammar Lv</strong> – average grammatical correctness (0-10 scale)
          </p>
          <p>
            <strong>Within 10 / 20 words</strong> – word-count distance from original
          </p>
        </div>
      </div>

      {allFolders.map((folder) => (
        <div key={folder} className="border rounded-md">
          <button
            onClick={() => toggleFolder(folder)}
            className="w-full px-4 py-3 flex items-center justify-between bg-muted/50 hover:bg-muted transition-colors"
          >
            <span className="font-medium">📁 {folder.replace("_", " ").toUpperCase()}</span>
            {expandedFolders.includes(folder) ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>

          {expandedFolders.includes(folder) && (
            <div className="p-4">
              <DetailedStatsTable stats={stats} folder={folder} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DetailedStatsTable({ stats, folder }) {
  const rows = createModelComparisonTable(stats, folder);
  const showStyleAdherence = rows.some((r) => r.styleAdherence !== null);

  if (rows.length === 0) {
    return <p className="text-center text-muted-foreground py-8">No data for this folder</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2 sticky left-0 bg-background z-10">Model</th>
            <th className="text-center p-2">Mode</th>
            <th className="text-center p-2">Drafts</th>
            <th className="text-center p-2">Paras</th>
            <th className="text-right p-2">Base GZ</th>
            <th className="text-right p-2">After GZ</th>
            <th className="text-right p-2">Δ GZ</th>
            <th className="text-right p-2">ZS GZ %</th>
            <th className="text-right p-2">Draft Δ %</th>
            <th className="text-right p-2">Para Δ %</th>
            <th className="text-right p-2">±10%</th>
            <th className="text-right p-2">±15%</th>
            <th className="text-right p-2">±20%</th>
            <th className="text-right p-2">% Longer</th>
            <th className="text-right p-2">% Shorter</th>
            <th className="text-right p-2">Quality %</th>
            {showStyleAdherence && <th className="text-right p-2">Style Score</th>}
            <th className="text-right p-2">Grammar</th>
            <th className="text-right p-2">Meaning</th>
            <th className="text-right p-2">Missing</th>
            <th className="text-right p-2">Cite Used %</th>
            <th className="text-right p-2">Cite OK %</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} className="border-b hover:bg-muted/50">
              <td className="p-2 font-medium sticky left-0 bg-background">{row.model}</td>
              <td className="text-center p-2">{row.mode}</td>
              <td className="text-center p-2">{row.drafts}</td>
              <td className="text-center p-2">{row.paragraphs}</td>
              <td className="text-right p-2">{formatMetric(row.baselineGz, 3)}</td>
              <td className="text-right p-2">{formatMetric(row.afterGz, 3)}</td>
              <td className={`text-right p-2 ${getDeltaColor(row.deltaGz)}`}>
                {formatMetric(row.deltaGz, 3)}
              </td>
              <td className={`text-right p-2 ${getZeroShotColor(row.zeroshotGz)}`}>
                {formatPct(row.zeroshotGz)}
              </td>
              <td className="text-right p-2">{formatPct(row.avgDraftDeltaPct)}</td>
              <td className="text-right p-2">{formatPct(row.avgParaDeltaPct)}</td>
              <td className={`text-right p-2 ${getLengthDeviationColor(row.lenWithin10Pct)}`}>
                {formatPct(row.lenWithin10Pct)}
              </td>
              <td className={`text-right p-2 ${getLengthDeviationColor(row.lenWithin15Pct)}`}>
                {formatPct(row.lenWithin15Pct)}
              </td>
              <td className={`text-right p-2 ${getLengthDeviationColor(row.lenWithin20Pct)}`}>
                {formatPct(row.lenWithin20Pct)}
              </td>
              <td className="text-right p-2">{formatPct(row.pctLonger)}</td>
              <td className="text-right p-2">{formatPct(row.pctShorter)}</td>
              <td className={`text-right p-2 ${getQualityColor(row.qualityPct)}`}>
                {formatPct(row.qualityPct)}
              </td>
              {showStyleAdherence && (
                <td className={`text-right p-2 ${getStyleAdherenceColor(row.styleAdherence)}`}>
                  {formatMetric(row.styleAdherence, 1)}
                </td>
              )}
              <td className={`text-right p-2 ${getGrammarColor(row.grammarLv)}`}>
                {formatMetric(row.grammarLv, 1)}
              </td>
              <td className={`text-right p-2 ${getMeaningLevelColor(row.sameMeaningLv)}`}>
                {formatMetric(row.sameMeaningLv, 1)}
              </td>
              <td className={`text-right p-2 ${getMissingInfoColor(row.missingInfoLv)}`}>
                {formatMetric(row.missingInfoLv, 1)}
              </td>
              <td className={`text-right p-2 ${getCitationColor(row.citationPreservedPct)}`}>
                {formatPct(row.citationPreservedPct)}
              </td>
              <td className={`text-right p-2 ${getCitationColor(row.citationExactPct)}`}>
                {formatPct(row.citationExactPct)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelPerformanceView({ stats }) {
  const rows = computeModelPerformance(stats);

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="font-medium mb-2">Model Performance Summary</h3>
        <p className="text-sm text-muted-foreground">
          Aggregated performance metrics across all folders and modes
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Model</th>
              <th className="text-center p-2">Mode</th>
              <th className="text-center p-2">Total Drafts</th>
              <th className="text-right p-2">Avg Δ GZ</th>
              <th className="text-right p-2">Avg Δ SP</th>
              <th className="text-right p-2">Zero-shot GZ</th>
              <th className="text-right p-2">Zero-shot SP</th>
              <th className="text-right p-2">Avg Quality</th>
              <th className="text-right p-2">Avg Grammar</th>
              <th className="text-center p-2">Folders</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} className="border-b hover:bg-muted/50">
                <td className="p-2 font-medium">{row.model}</td>
                <td className="text-center p-2">{row.mode}</td>
                <td className="text-center p-2">{row.totalDrafts}</td>
                <td className={`text-right p-2 ${getDeltaColor(row.avgDeltaGz)}`}>
                  {formatMetric(row.avgDeltaGz, 3)}
                </td>
                <td className={`text-right p-2 ${getDeltaColor(row.avgDeltaSp)}`}>
                  {formatMetric(row.avgDeltaSp, 3)}
                </td>
                <td className={`text-right p-2 ${getZeroShotColor(row.zeroshotGz)}`}>
                  {formatPct(row.zeroshotGz)}
                </td>
                <td className={`text-right p-2 ${getZeroShotColor(row.zeroshotSp)}`}>
                  {formatPct(row.zeroshotSp)}
                </td>
                <td className={`text-right p-2 ${getQualityColor(row.avgQuality)}`}>
                  {formatPct(row.avgQuality)}
                </td>
                <td className={`text-right p-2 ${getGrammarColor(row.avgGrammar)}`}>
                  {formatMetric(row.avgGrammar, 1)}
                </td>
                <td className="text-center p-2">{row.folders}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ExtendedStatsView({ stats }) {
  const rows = buildExtendedStats(stats);

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="font-medium mb-2">Extended Statistical Analysis</h3>
        <p className="text-sm text-muted-foreground">
          Descriptive statistics: Min, P25, Median, Mean, P75, Max
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Folder</th>
              <th className="text-left p-2">Model</th>
              <th className="text-center p-2">Mode</th>
              <th className="text-right p-2" colSpan={6}>
                GPTZero
              </th>
              <th className="text-right p-2" colSpan={6}>
                Quality %
              </th>
              <th className="text-right p-2" colSpan={6}>
                Grammar Level
              </th>
            </tr>
            <tr className="border-b text-muted-foreground">
              <th className="p-2"></th>
              <th className="p-2"></th>
              <th className="p-2"></th>
              <th className="text-right p-2">Min</th>
              <th className="text-right p-2">P25</th>
              <th className="text-right p-2">Med</th>
              <th className="text-right p-2">Mean</th>
              <th className="text-right p-2">P75</th>
              <th className="text-right p-2">Max</th>
              <th className="text-right p-2">Min</th>
              <th className="text-right p-2">P25</th>
              <th className="text-right p-2">Med</th>
              <th className="text-right p-2">Mean</th>
              <th className="text-right p-2">P75</th>
              <th className="text-right p-2">Max</th>
              <th className="text-right p-2">Min</th>
              <th className="text-right p-2">P25</th>
              <th className="text-right p-2">Med</th>
              <th className="text-right p-2">Mean</th>
              <th className="text-right p-2">P75</th>
              <th className="text-right p-2">Max</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx} className="border-b hover:bg-muted/50">
                <td className="p-2">{row.folder}</td>
                <td className="p-2 font-medium">{row.model}</td>
                <td className="text-center p-2">{row.mode}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.min, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.p25, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.median, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.mean, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.p75, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.gptZero.max, 3)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.min, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.p25, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.median, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.mean, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.p75, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.quality.max, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.min, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.p25, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.median, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.mean, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.p75, 1)}</td>
                <td className="text-right p-2">{formatMetric(row.grammarLv.max, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FolderSummaryView({ stats }) {
  const folderSummaries = Object.entries(stats).map(([folder, models]) => {
    const allRows = [];
    Object.entries(models).forEach(([model, modes]) => {
      Object.entries(modes).forEach(([mode, data]) => {
        allRows.push(data);
      });
    });

    const totalDrafts = allRows.reduce((sum, d) => sum + (d.draft_count || 0), 0);
    const avgGzDelta =
      allRows.reduce((sum, d) => sum + (d.deltas?.gptzero || 0), 0) / allRows.length;
    const avgZsGz =
      allRows.reduce((sum, d) => sum + (d.zero_shot_success?.gptzero || 0), 0) / allRows.length;
    const avgQuality =
      allRows.reduce((sum, d) => {
        const q = d.quality ? Object.values(d.quality).reduce((a, b) => a + b, 0) / Object.values(d.quality).length : 0;
        return sum + q;
      }, 0) / allRows.length;

    return {
      folder,
      totalDrafts,
      models: Object.keys(models).length,
      avgGzDelta,
      avgZsGz,
      avgQuality,
    };
  });

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="font-medium mb-2">Folder Performance Summary</h3>
        <p className="text-sm text-muted-foreground">
          Aggregated metrics by document folder
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2">Folder</th>
              <th className="text-center p-2">Models</th>
              <th className="text-center p-2">Total Drafts</th>
              <th className="text-right p-2">Avg Δ GZ</th>
              <th className="text-right p-2">Avg Zero-shot GZ</th>
              <th className="text-right p-2">Avg Quality</th>
            </tr>
          </thead>
          <tbody>
            {folderSummaries.map((row) => (
              <tr key={row.folder} className="border-b hover:bg-muted/50">
                <td className="p-2 font-medium">📁 {row.folder.replace("_", " ").toUpperCase()}</td>
                <td className="text-center p-2">{row.models}</td>
                <td className="text-center p-2">{row.totalDrafts}</td>
                <td className={`text-right p-2 ${getDeltaColor(row.avgGzDelta)}`}>
                  {formatMetric(row.avgGzDelta, 3)}
                </td>
                <td className={`text-right p-2 ${getZeroShotColor(row.avgZsGz)}`}>
                  {formatPct(row.avgZsGz)}
                </td>
                <td className={`text-right p-2 ${getQualityColor(row.avgQuality)}`}>
                  {formatPct(row.avgQuality)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DistributionsView({ stats }) {
  return (
    <div className="space-y-4">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-md">
        <h3 className="font-medium mb-2">Score & Word-count Distributions</h3>
        <p className="text-sm text-muted-foreground">
          Distribution charts will be available in a future update
        </p>
      </div>
      <p className="text-center text-muted-foreground py-8">
        Chart visualizations coming soon
      </p>
    </div>
  );
}
