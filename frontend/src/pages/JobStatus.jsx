import { useState, useEffect, useRef } from "react";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { Progress } from "../components/ui/Progress";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/Alert";
import { formatTimestamp, formatDuration, getStatusColor } from "../lib/utils";
import { RefreshCw, XCircle, CheckCircle, Clock, Loader2 } from "lucide-react";

export function JobStatus() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  const loadJobs = async () => {
    try {
      setError(null);
      const data = await api.listJobs(50);
      setJobs(data);
    } catch (err) {
      setError(err.message || "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  };

  const handleCancelJob = async (jobId) => {
    try {
      await api.cancelJob(jobId);
      await loadJobs();
    } catch (err) {
      alert(`Failed to cancel job: ${err.message}`);
    }
  };

  useEffect(() => {
    loadJobs();

    // Set up WebSocket for real-time updates
    let reconnectTimeout;
    const connectWebSocket = () => {
      try {
        const ws = api.createWebSocket();
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("WebSocket connected");
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === "job_update") {
            setJobs((prevJobs) => {
              const index = prevJobs.findIndex((j) => j.job_id === data.job_id);
              if (index >= 0) {
                const newJobs = [...prevJobs];
                newJobs[index] = { ...newJobs[index], ...data.data };
                return newJobs;
              } else {
                return [data.data, ...prevJobs];
              }
            });
          }
        };

        ws.onerror = (error) => {
          console.warn("WebSocket error (will retry):", error);
        };

        ws.onclose = () => {
          console.log("WebSocket connection closed, will reconnect in 5s");
          // Reconnect after 5 seconds
          reconnectTimeout = setTimeout(() => {
            connectWebSocket();
          }, 5000);
        };
      } catch (err) {
        console.error("Failed to create WebSocket:", err);
      }
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const runningJobs = jobs.filter((j) => j.status === "running");
  const recentJobs = jobs.filter((j) => j.status !== "running").slice(0, 10);

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
          <h1 className="text-3xl font-bold">Job Status</h1>
          <p className="text-muted-foreground mt-1">Monitor active and recent jobs</p>
        </div>
        <Button onClick={loadJobs} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Running Jobs */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Running Jobs ({runningJobs.length})</h2>
        {runningJobs.length === 0 ? (
          <Card>
            <CardContent className="p-6">
              <p className="text-center text-muted-foreground">No running jobs</p>
            </CardContent>
          </Card>
        ) : (
          runningJobs.map((job) => (
            <JobCard key={job.job_id} job={job} onCancel={handleCancelJob} />
          ))
        )}
      </div>

      {/* Recent Jobs */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Recent Jobs</h2>
        {recentJobs.length === 0 ? (
          <Card>
            <CardContent className="p-6">
              <p className="text-center text-muted-foreground">No recent jobs</p>
            </CardContent>
          </Card>
        ) : (
          recentJobs.map((job) => (
            <JobCard key={job.job_id} job={job} />
          ))
        )}
      </div>
    </div>
  );
}

function JobCard({ job, onCancel }) {
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState([]);
  const [loadingLogs, setLoadingLogs] = useState(false);

  const progress = job.total_docs > 0 ? (job.processed_docs / job.total_docs) * 100 : 0;

  const loadLogs = async () => {
    if (showLogs) {
      setShowLogs(false);
      return;
    }

    setLoadingLogs(true);
    try {
      const data = await api.getJobLogs(job.job_id);
      setLogs(data.logs || []);
      setShowLogs(true);
    } catch (err) {
      alert(`Failed to load logs: ${err.message}`);
    } finally {
      setLoadingLogs(false);
    }
  };

  const StatusIcon = {
    pending: Clock,
    running: Loader2,
    completed: CheckCircle,
    failed: XCircle,
    cancelled: XCircle,
  }[job.status] || Clock;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-lg">{job.run_name}</CardTitle>
            <CardDescription>Job ID: {job.job_id}</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={getStatusColor(job.status)}>
              <StatusIcon className={`h-3 w-3 mr-1 ${job.status === "running" ? "animate-spin" : ""}`} />
              {job.status}
            </Badge>
            {job.status === "running" && onCancel && (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => onCancel(job.job_id)}
              >
                Cancel
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Created</p>
            <p className="font-medium">{formatTimestamp(job.created_at)}</p>
          </div>
          {job.started_at && (
            <div>
              <p className="text-muted-foreground">Started</p>
              <p className="font-medium">{formatTimestamp(job.started_at)}</p>
            </div>
          )}
          {job.completed_at && (
            <div>
              <p className="text-muted-foreground">Completed</p>
              <p className="font-medium">{formatTimestamp(job.completed_at)}</p>
            </div>
          )}
          {job.started_at && job.completed_at && (
            <div>
              <p className="text-muted-foreground">Duration</p>
              <p className="font-medium">{formatDuration(job.completed_at - job.started_at)}</p>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">
              {job.processed_docs} / {job.total_docs} documents
            </span>
          </div>
          <Progress value={progress} />
          {job.current_doc && (
            <p className="text-sm text-muted-foreground">
              Current: {job.current_doc}
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Models</p>
            <p className="font-medium">{job.models?.join(", ") || "N/A"}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Iterations</p>
            <p className="font-medium">{job.iterations || "N/A"}</p>
          </div>
        </div>

        {job.error && (
          <Alert variant="destructive">
            <AlertDescription>{job.error}</AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={loadLogs}
            disabled={loadingLogs}
          >
            {loadingLogs ? "Loading..." : showLogs ? "Hide Logs" : "Show Logs"}
          </Button>
        </div>

        {showLogs && (
          <div className="bg-muted rounded-md p-4 max-h-64 overflow-y-auto">
            <pre className="text-xs font-mono whitespace-pre-wrap">
              {logs.length > 0 ? logs.join("\n") : "No logs available"}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
