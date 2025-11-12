import { useState, useEffect, useMemo } from "react";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Select } from "../components/ui/Select";
import { Badge } from "../components/ui/Badge";
import { Alert, AlertDescription, AlertTitle, AlertIcons } from "../components/ui/Alert";
import { formatNumber, formatPercent, downloadJSON } from "../lib/utils";
import { Loader2, Download, BarChart3, TrendingUp, TrendingDown } from "lucide-react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
} from "@tanstack/react-table";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export function BenchmarkAnalysis() {
  const [runs, setRuns] = useState([]);
  const [selectedRuns, setSelectedRuns] = useState([]);
  const [runData, setRunData] = useState({});
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(false);
  const [error, setError] = useState(null);
  const [mergeRuns, setMergeRuns] = useState(false);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      setError(null);
      const data = await api.listRuns();
      setRuns(data);
    } catch (err) {
      setError(err.message || "Failed to load runs");
    } finally {
      setLoading(false);
    }
  };

  const loadRunData = async (runName) => {
    if (runData[runName]) return; // Already loaded

    try {
      const data = await api.loadRun(runName);
      setRunData((prev) => ({ ...prev, [runName]: data }));
    } catch (err) {
      alert(`Failed to load run ${runName}: ${err.message}`);
    }
  };

  const handleRunSelection = async (runName) => {
    const newSelection = selectedRuns.includes(runName)
      ? selectedRuns.filter((r) => r !== runName)
      : [...selectedRuns, runName];

    setSelectedRuns(newSelection);

    // Load run data if selecting
    if (!selectedRuns.includes(runName)) {
      await loadRunData(runName);
    }
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
    const exportData = selectedRuns.map((runName) => ({
      run_name: runName,
      data: runData[runName],
    }));
    downloadJSON(exportData, `benchmark_export_${Date.now()}.json`);
  };

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
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {runs.map((run) => (
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

            <div className="flex gap-3">
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
              <Button variant="outline" onClick={handleDownloadData}>
                <Download className="h-4 w-4 mr-2" />
                Download Data
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics Results */}
      {statistics && (
        <StatisticsDisplay statistics={statistics} selectedRuns={selectedRuns} />
      )}

      {/* Run Data Display */}
      {selectedRuns.length > 0 && !statistics && (
        <div className="space-y-6">
          {selectedRuns.map((runName) => (
            runData[runName] && (
              <RunDataDisplay key={runName} runName={runName} data={runData[runName]} />
            )
          ))}
        </div>
      )}
    </div>
  );
}

function StatisticsDisplay({ statistics, selectedRuns }) {
  if (!statistics || !statistics.models) return null;

  const chartData = Object.entries(statistics.models).map(([model, stats]) => ({
    model: model.length > 15 ? model.substring(0, 15) + "..." : model,
    mean: stats.mean_ai_score || 0,
    median: stats.median_ai_score || 0,
    zeroshot: (stats.zeroshot_success || 0) * 100,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Statistical Summary</CardTitle>
          <CardDescription>
            Analysis of {selectedRuns.length} run{selectedRuns.length > 1 ? "s" : ""}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Chart */}
            <div>
              <h3 className="text-sm font-medium mb-4">AI Score Distribution by Model</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mean" fill="#3b82f6" name="Mean AI Score" />
                  <Bar dataKey="median" fill="#10b981" name="Median AI Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Zero-shot Success Chart */}
            <div>
              <h3 className="text-sm font-medium mb-4">Zero-shot Success Rate (%)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="zeroshot" fill="#8b5cf6" name="Zero-shot Success %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Statistics Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">Model</th>
                    <th className="text-right p-2">Mean</th>
                    <th className="text-right p-2">Median</th>
                    <th className="text-right p-2">Std Dev</th>
                    <th className="text-right p-2">P25</th>
                    <th className="text-right p-2">P75</th>
                    <th className="text-right p-2">Zero-shot %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(statistics.models).map(([model, stats]) => (
                    <tr key={model} className="border-b hover:bg-muted/50">
                      <td className="p-2 font-medium">{model}</td>
                      <td className="text-right p-2">{formatNumber(stats.mean_ai_score)}</td>
                      <td className="text-right p-2">{formatNumber(stats.median_ai_score)}</td>
                      <td className="text-right p-2">{formatNumber(stats.std_ai_score)}</td>
                      <td className="text-right p-2">{formatNumber(stats.p25_ai_score)}</td>
                      <td className="text-right p-2">{formatNumber(stats.p75_ai_score)}</td>
                      <td className="text-right p-2">{formatPercent(stats.zeroshot_success, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function RunDataDisplay({ runName, data }) {
  const [sorting, setSorting] = useState([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const columns = useMemo(
    () => [
      {
        accessorKey: "doc_name",
        header: "Document",
        cell: (info) => <span className="font-medium">{info.getValue()}</span>,
      },
      {
        accessorKey: "folder",
        header: "Folder",
        cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
      },
      {
        accessorKey: "model",
        header: "Model",
      },
      {
        accessorKey: "iteration",
        header: "Iter",
        cell: (info) => <span className="text-center">{info.getValue()}</span>,
      },
      {
        accessorKey: "ai_score",
        header: "AI Score",
        cell: (info) => {
          const value = info.getValue();
          const color = value < 0.1 ? "text-green-600" : value < 0.3 ? "text-yellow-600" : "text-red-600";
          return <span className={color}>{formatNumber(value, 3)}</span>;
        },
      },
      {
        accessorKey: "quality_score",
        header: "Quality",
        cell: (info) => {
          const value = info.getValue();
          return value !== null ? formatNumber(value, 1) : "N/A";
        },
      },
    ],
    []
  );

  const tableData = useMemo(() => {
    if (!data || !data.docs) return [];

    const rows = [];
    data.docs.forEach((doc) => {
      if (doc.models) {
        Object.entries(doc.models).forEach(([model, modelData]) => {
          if (modelData.iterations) {
            modelData.iterations.forEach((iter, idx) => {
              rows.push({
                doc_name: doc.doc_name,
                folder: doc.folder,
                model,
                iteration: idx + 1,
                ai_score: iter.para_ai_score || iter.doc_ai_score || 0,
                quality_score: iter.para_quality_score || iter.doc_quality_score,
              });
            });
          }
        });
      }
    });
    return rows;
  }, [data]);

  const table = useReactTable({
    data: tableData,
    columns,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>{runName}</CardTitle>
        <CardDescription>
          {data.docs?.length || 0} documents, {data.models?.length || 0} models
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <input
            type="text"
            placeholder="Search..."
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="px-3 py-2 border rounded-md w-full max-w-sm"
          />

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                {table.getHeaderGroups().map((headerGroup) => (
                  <tr key={headerGroup.id} className="border-b">
                    {headerGroup.headers.map((header) => (
                      <th
                        key={header.id}
                        className="text-left p-2 cursor-pointer hover:bg-muted/50"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() && (
                          <span className="ml-2">
                            {header.column.getIsSorted() === "asc" ? <TrendingUp className="h-3 w-3 inline" /> : <TrendingDown className="h-3 w-3 inline" />}
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.slice(0, 100).map((row) => (
                  <tr key={row.id} className="border-b hover:bg-muted/50">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="p-2">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {table.getRowModel().rows.length > 100 && (
            <p className="text-sm text-muted-foreground text-center">
              Showing first 100 rows of {table.getRowModel().rows.length}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
