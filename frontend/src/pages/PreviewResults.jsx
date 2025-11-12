import { useState, useEffect } from "react";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/Alert";
import { formatNumber, formatPercent } from "../lib/utils";
import { Loader2, TrendingUp, TrendingDown, Minus } from "lucide-react";

export function PreviewResults() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [runData, setRunData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      setError(null);
      const data = await api.listRuns();
      // API returns { runs: [...], total: N }
      setRuns(data.runs || []);
    } catch (err) {
      setError(err.message || "Failed to load runs");
    } finally {
      setLoading(false);
    }
  };

  const loadRunData = async (runName) => {
    setLoadingRun(true);
    try {
      setError(null);
      const data = await api.loadRun(runName);
      setRunData(data);
      setSelectedRun(runName);
    } catch (err) {
      setError(err.message || "Failed to load run data");
    } finally {
      setLoadingRun(false);
    }
  };

  // Quick preview stats calculation
  const computeQuickStats = () => {
    if (!runData || !runData.docs) return null;

    const modelStats = {};

    runData.docs.forEach((doc) => {
      if (doc.models) {
        Object.entries(doc.models).forEach(([model, modelData]) => {
          if (!modelStats[model]) {
            modelStats[model] = {
              scores: [],
              qualityScores: [],
              zeroShotCount: 0,
              totalCount: 0,
            };
          }

          if (modelData.iterations) {
            modelData.iterations.forEach((iter) => {
              const score = iter.para_ai_score || iter.doc_ai_score;
              if (score !== null && score !== undefined) {
                modelStats[model].scores.push(score);
                modelStats[model].totalCount++;
                if (score <= 0.1) {
                  modelStats[model].zeroShotCount++;
                }
              }

              const quality = iter.para_quality_score || iter.doc_quality_score;
              if (quality !== null && quality !== undefined) {
                modelStats[model].qualityScores.push(quality);
              }
            });
          }
        });
      }
    });

    // Calculate averages
    const results = Object.entries(modelStats).map(([model, stats]) => {
      const avgScore = stats.scores.length > 0
        ? stats.scores.reduce((a, b) => a + b, 0) / stats.scores.length
        : 0;

      const avgQuality = stats.qualityScores.length > 0
        ? stats.qualityScores.reduce((a, b) => a + b, 0) / stats.qualityScores.length
        : 0;

      const zeroShotRate = stats.totalCount > 0
        ? stats.zeroShotCount / stats.totalCount
        : 0;

      return {
        model,
        avgScore,
        avgQuality,
        zeroShotRate,
        sampleCount: stats.totalCount,
      };
    });

    return results.sort((a, b) => a.avgScore - b.avgScore);
  };

  const stats = computeQuickStats();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Preview Results</h1>
          <p className="text-muted-foreground mt-1">Quick screening and model comparison</p>
        </div>
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
          <CardTitle>Select Run</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {runs.map((run) => (
              <button
                key={run.name}
                onClick={() => loadRunData(run.name)}
                className={`p-3 border rounded-md text-left transition-colors ${
                  selectedRun === run.name
                    ? "bg-primary text-primary-foreground border-primary"
                    : "hover:bg-accent hover:text-accent-foreground"
                }`}
                disabled={loadingRun}
              >
                <div className="font-medium">{run.name}</div>
                <div className="text-xs opacity-75">
                  {new Date(run.timestamp * 1000).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Loading State */}
      {loadingRun && (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}

      {/* Quick Stats */}
      {stats && !loadingRun && (
        <Card>
          <CardHeader>
            <CardTitle>Model Performance Quick View</CardTitle>
            <CardDescription>
              Showing {stats.length} models sorted by average AI score (lower is better)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {stats.map((stat, index) => (
                <div
                  key={stat.model}
                  className="flex items-center gap-4 p-4 border rounded-md hover:bg-muted/50 transition-colors"
                >
                  <div className="flex-shrink-0 w-8 text-center">
                    <span className="text-lg font-bold text-muted-foreground">#{index + 1}</span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{stat.model}</div>
                    <div className="text-xs text-muted-foreground">
                      {stat.sampleCount} samples
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 flex-shrink-0">
                    <div className="text-center">
                      <div className="text-xs text-muted-foreground mb-1">Avg AI Score</div>
                      <div className={`font-semibold ${
                        stat.avgScore < 0.1 ? "text-green-600" :
                        stat.avgScore < 0.3 ? "text-yellow-600" :
                        "text-red-600"
                      }`}>
                        {formatNumber(stat.avgScore, 3)}
                      </div>
                    </div>

                    <div className="text-center">
                      <div className="text-xs text-muted-foreground mb-1">Zero-shot %</div>
                      <div className="font-semibold">
                        {formatPercent(stat.zeroShotRate, 0)}
                      </div>
                    </div>

                    <div className="text-center">
                      <div className="text-xs text-muted-foreground mb-1">Avg Quality</div>
                      <div className="font-semibold">
                        {stat.avgQuality > 0 ? formatNumber(stat.avgQuality, 1) : "N/A"}
                      </div>
                    </div>
                  </div>

                  <div className="flex-shrink-0">
                    {index === 0 ? (
                      <Badge variant="secondary" className="bg-green-500/10 text-green-700 border-green-500/20">
                        <TrendingDown className="h-3 w-3 mr-1" />
                        Best
                      </Badge>
                    ) : index < 3 ? (
                      <Badge variant="outline">
                        <TrendingDown className="h-3 w-3 mr-1" />
                        Top 3
                      </Badge>
                    ) : index === stats.length - 1 ? (
                      <Badge variant="secondary" className="bg-red-500/10 text-red-700 border-red-500/20">
                        <TrendingUp className="h-3 w-3 mr-1" />
                        Worst
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="opacity-50">
                        <Minus className="h-3 w-3" />
                      </Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {!loadingRun && !stats && selectedRun && (
        <Card>
          <CardContent className="p-6">
            <p className="text-center text-muted-foreground">No data available for this run</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
