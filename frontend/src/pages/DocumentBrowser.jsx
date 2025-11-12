import { useState, useEffect } from "react";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Select } from "../components/ui/Select";
import { Badge } from "../components/ui/Badge";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/Alert";
import { formatNumber } from "../lib/utils";
import { Loader2, FileText, ChevronRight } from "lucide-react";

export function DocumentBrowser() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docDetails, setDocDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      setError(null);
      const data = await api.listRuns();
      setRuns(data);
      if (data.length > 0 && !selectedRun) {
        setSelectedRun(data[0].name);
        loadDocuments(data[0].name);
      }
    } catch (err) {
      setError(err.message || "Failed to load runs");
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async (runName) => {
    try {
      setError(null);
      const data = await api.getDocuments(runName);
      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message || "Failed to load documents");
    }
  };

  const loadDocumentDetails = async (runName, docName) => {
    setLoadingDoc(true);
    try {
      setError(null);
      const data = await api.getDocument(runName, docName);
      setDocDetails(data);
      setSelectedDoc(docName);
    } catch (err) {
      setError(err.message || "Failed to load document details");
    } finally {
      setLoadingDoc(false);
    }
  };

  const handleRunChange = (runName) => {
    setSelectedRun(runName);
    setSelectedDoc(null);
    setDocDetails(null);
    loadDocuments(runName);
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
          <h1 className="text-3xl font-bold">Document Browser</h1>
          <p className="text-muted-foreground mt-1">Browse and analyze individual documents</p>
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
          <Select value={selectedRun} onChange={(e) => handleRunChange(e.target.value)}>
            <option value="">Select a run...</option>
            {runs.map((run) => (
              <option key={run.name} value={run.name}>
                {run.name} ({new Date(run.timestamp * 1000).toLocaleDateString()})
              </option>
            ))}
          </Select>
        </CardContent>
      </Card>

      {/* Document List & Details */}
      {selectedRun && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Document List */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>Documents</CardTitle>
              <CardDescription>{documents.length} documents</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {documents.map((doc) => (
                  <button
                    key={doc}
                    onClick={() => loadDocumentDetails(selectedRun, doc)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors flex items-center justify-between ${
                      selectedDoc === doc
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-accent hover:text-accent-foreground"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      <span className="truncate">{doc}</span>
                    </span>
                    <ChevronRight className="h-4 w-4" />
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Document Details */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Document Details</CardTitle>
              {selectedDoc && <CardDescription>{selectedDoc}</CardDescription>}
            </CardHeader>
            <CardContent>
              {loadingDoc ? (
                <div className="flex items-center justify-center h-64">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : docDetails ? (
                <DocumentDetailsView data={docDetails} />
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  Select a document to view details
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

function DocumentDetailsView({ data }) {
  if (!data || !data.models) {
    return <p className="text-muted-foreground">No data available</p>;
  }

  return (
    <div className="space-y-6">
      {/* Document Info */}
      <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-md">
        <div>
          <p className="text-sm text-muted-foreground">Document</p>
          <p className="font-medium">{data.doc_name}</p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Folder</p>
          <p className="font-medium">{data.folder}</p>
        </div>
      </div>

      {/* Model Results */}
      <div className="space-y-4">
        <h3 className="font-semibold">Model Results</h3>
        {Object.entries(data.models).map(([model, modelData]) => (
          <div key={model} className="border rounded-md p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">{model}</h4>
              <Badge>{modelData.iterations?.length || 0} iterations</Badge>
            </div>

            {/* Iterations */}
            <div className="space-y-2">
              {modelData.iterations?.map((iter, idx) => (
                <div key={idx} className="bg-muted p-3 rounded-md">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Iteration {idx + 1}</span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground">Para AI Score</p>
                      <p className="font-medium">
                        {iter.para_ai_score !== null ? formatNumber(iter.para_ai_score, 3) : "N/A"}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Doc AI Score</p>
                      <p className="font-medium">
                        {iter.doc_ai_score !== null ? formatNumber(iter.doc_ai_score, 3) : "N/A"}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Para Quality</p>
                      <p className="font-medium">
                        {iter.para_quality_score !== null ? formatNumber(iter.para_quality_score, 1) : "N/A"}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Doc Quality</p>
                      <p className="font-medium">
                        {iter.doc_quality_score !== null ? formatNumber(iter.doc_quality_score, 1) : "N/A"}
                      </p>
                    </div>
                  </div>

                  {/* Show rewritten text if available */}
                  {iter.para_rewritten && (
                    <details className="mt-3">
                      <summary className="cursor-pointer text-sm text-primary hover:underline">
                        View Rewritten Text
                      </summary>
                      <div className="mt-2 p-2 bg-background rounded text-xs max-h-40 overflow-y-auto whitespace-pre-wrap">
                        {iter.para_rewritten}
                      </div>
                    </details>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
