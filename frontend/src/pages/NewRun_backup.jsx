import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Select } from "../components/ui/Select";
import { Alert, AlertDescription, AlertTitle, AlertIcons } from "../components/ui/Alert";
import { PlayCircle, Loader2 } from "lucide-react";

// Model registry (sync with backend)
const MODEL_LIST = [
  // Vanilla OpenAI
  "gpt-4.1", "gpt-4.1-mini", "gpt-4o",
  // Claude
  "claude-sonnet-4", "claude-sonnet-3.7", "claude-haiku-3.5",
  // Gemini
  "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
  // Fine-tuned models
  "gpt-4o-old-model", "gpt-4o-hum30raw", "gpt-4o-hum40naive",
  "gpt-4.1-hum30start", "gpt-4.1-hum40naive", "gpt-4.1-mini-hum40naive",
  "gpt-4.1-hum30raw-fix1", "gpt-4o-hum30raw-fix1", "gpt-4o-hum30raw-fix2",
  "gpt-4.1-hum30raw-fix2", "gpt-4o-hum30raw-retrain1", "gpt-4o-mini-hum30raw-fix1",
  // Dynamic models (gpt-4o-mini)
  "min-e3-b8-m10-v10", "raw-e5-b16-m08-v10", "cmp-e4-b24-m12-v15",
  "min-e6-b16-m05-v20", "rubx-e8-b32-m03-v15", "raw-e3-b8-m15-v10-mini",
  // Dynamic models (gpt-4.1-mini)
  "min-e3-b8-m10-v10-d2", "cmp-e5-b16-m08-v15", "raw-e4-b24-m06-v20",
  "rdesc-e6-b20-m10-v10", "min-e8-b32-m04-v20", "cmp-e3-b12-m12-v12",
  // Dynamic models (gpt-4.1)
  "min-e3-b6-m08-v15", "raw-e4-b8-m06-v10", "cmp-e5-b8-m05-v20",
  "rchi-e6-b12-m04-v15", "min-e8-b16-m03-v20", "raw-e3-b10-m10-v10",
  // Dynamic models (gpt-4.1-nano)
  "min-e3-b32-m10-v10-nano", "raw-e5-b48-m08-v15", "cmp-e6-b64-m06-v20",
  "min-e4-b40-m12-v12", "rich-e8-b64-m04-v20", "raw-e3-b24-m15-v10-nano",
  // Dynamic models (gpt-4o)
  "min-e3-b4-m06-v15", "raw-e4-b6-m05-v10", "cmp-e5-b6-m04-v20",
  "rich-e6-b8-m03-v20", "min-e3-b5-m10-v10-4o", "raw-e8-b8-m025-v20",
];

const FOLDERS = {
  "AI texts": "data/ai_texts",
  "Human texts": "data/human_texts",
  "AI paragraphs": "data/ai_paras",
  "Human paragraphs": "data/human_paras",
};

export function NewRun() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const [formData, setFormData] = useState({
    runName: `run_${Date.now()}`,
    folders: ["data/ai_texts"],
    models: ["gpt-4.1"],
    iterations: 5,
    docCounts: {},
    includeDocMode: true,
    useGptzero: true,
    useSapling: true,
  });

  const handleFolderToggle = (folder) => {
    setFormData((prev) => {
      const newFolders = prev.folders.includes(folder)
        ? prev.folders.filter((f) => f !== folder)
        : [...prev.folders, folder];
      return { ...prev, folders: newFolders };
    });
  };

  const handleModelToggle = (model) => {
    setFormData((prev) => {
      const newModels = prev.models.includes(model)
        ? prev.models.filter((m) => m !== model)
        : [...prev.models, model];
      return { ...prev, models: newModels };
    });
  };

  const handleDocCountChange = (folder, count) => {
    setFormData((prev) => ({
      ...prev,
      docCounts: {
        ...prev.docCounts,
        [folder]: count ? parseInt(count) : undefined,
      },
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    setLoading(true);

    // Validation
    if (!formData.runName) {
      setError("Please provide a run name");
      setLoading(false);
      return;
    }

    if (formData.folders.length === 0) {
      setError("Please select at least one folder");
      setLoading(false);
      return;
    }

    if (formData.models.length === 0) {
      setError("Please select at least one model");
      setLoading(false);
      return;
    }

    try {
      const jobData = {
        run_name: formData.runName,
        folders: formData.folders,
        models: formData.models,
        iterations: formData.iterations,
        doc_counts: formData.docCounts,
        include_doc_mode: formData.includeDocMode,
        use_gptzero: formData.useGptzero,
        use_sapling: formData.useSapling,
      };

      const response = await api.createJob(jobData);
      setSuccess(`Job created successfully! Job ID: ${response.job_id}`);

      // Navigate to job status after a short delay
      setTimeout(() => {
        navigate("/");
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to create job");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-bold">New Benchmark Run</h1>
        <p className="text-muted-foreground mt-1">Create and configure a new humanization benchmark</p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertIcons.error className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert variant="success">
          <AlertIcons.success className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Run Name */}
        <Card>
          <CardHeader>
            <CardTitle>Run Configuration</CardTitle>
            <CardDescription>Basic settings for the benchmark run</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="runName" className="text-sm font-medium">
                Run Name
              </label>
              <Input
                id="runName"
                value={formData.runName}
                onChange={(e) => setFormData({ ...formData, runName: e.target.value })}
                placeholder="Enter run name"
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="iterations" className="text-sm font-medium">
                Iterations per Document
              </label>
              <Input
                id="iterations"
                type="number"
                min="1"
                max="20"
                value={formData.iterations}
                onChange={(e) => setFormData({ ...formData, iterations: parseInt(e.target.value) })}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                Number of humanization attempts per document
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Folders */}
        <Card>
          <CardHeader>
            <CardTitle>Document Folders</CardTitle>
            <CardDescription>Select which document folders to process</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(FOLDERS).map(([name, path]) => (
              <div key={path} className="flex items-start gap-3 p-3 border rounded-md">
                <input
                  type="checkbox"
                  id={path}
                  checked={formData.folders.includes(path)}
                  onChange={() => handleFolderToggle(path)}
                  disabled={loading}
                  className="mt-1"
                />
                <div className="flex-1">
                  <label htmlFor={path} className="text-sm font-medium cursor-pointer">
                    {name}
                  </label>
                  <p className="text-xs text-muted-foreground">{path}</p>
                  {formData.folders.includes(path) && (
                    <div className="mt-2">
                      <Input
                        type="number"
                        min="1"
                        placeholder="All documents (leave empty)"
                        value={formData.docCounts[path] || ""}
                        onChange={(e) => handleDocCountChange(path, e.target.value)}
                        disabled={loading}
                        className="max-w-xs"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Limit number of documents (optional)
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Models */}
        <Card>
          <CardHeader>
            <CardTitle>Humanization Models</CardTitle>
            <CardDescription>Select which models to test ({formData.models.length} selected)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {MODEL_LIST.map((model) => (
                <div key={model} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id={model}
                    checked={formData.models.includes(model)}
                    onChange={() => handleModelToggle(model)}
                    disabled={loading}
                  />
                  <label htmlFor={model} className="text-sm cursor-pointer">
                    {model}
                  </label>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Detection & Analysis Options */}
        <Card>
          <CardHeader>
            <CardTitle>Detection & Analysis Options</CardTitle>
            <CardDescription>Configure detection and quality checking</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="includeDocMode"
                checked={formData.includeDocMode}
                onChange={(e) => setFormData({ ...formData, includeDocMode: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="includeDocMode" className="text-sm font-medium cursor-pointer">
                Include Document-Level Rewriting
              </label>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="useGptzero"
                checked={formData.useGptzero}
                onChange={(e) => setFormData({ ...formData, useGptzero: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="useGptzero" className="text-sm font-medium cursor-pointer">
                Use GPTZero Detection
              </label>
            </div>

            <div className="flex items-center gap-3">
              <input
                type="checkbox"
                id="useSapling"
                checked={formData.useSapling}
                onChange={(e) => setFormData({ ...formData, useSapling: e.target.checked })}
                disabled={loading}
              />
              <label htmlFor="useSapling" className="text-sm font-medium cursor-pointer">
                Use Sapling Detection
              </label>
            </div>
          </CardContent>
        </Card>

        {/* Submit */}
        <div className="flex gap-3">
          <Button type="submit" disabled={loading} className="min-w-32">
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4 mr-2" />
                Start Run
              </>
            )}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => navigate("/")}
            disabled={loading}
          >
            Cancel
          </Button>
        </div>
      </form>
    </div>
  );
}
