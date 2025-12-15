import { NavLink, useNavigate } from "react-router-dom";
import { cn } from "../lib/utils";
import {
  LayoutDashboard,
  PlayCircle,
  BarChart3,
  FileText,
  Activity,
  LogOut
} from "lucide-react";

const navigation = [
  { name: "Job Status", href: "/", icon: Activity },
  { name: "New Run", href: "/new-run", icon: PlayCircle },
  { name: "Benchmark Analysis", href: "/benchmark-analysis", icon: BarChart3 },
  { name: "Document Browser", href: "/document-browser", icon: FileText },
  { name: "Preview Results", href: "/preview-results", icon: LayoutDashboard },
  { name: "Writing Profile Lab", href: "/writing-profile", icon: FileText },
];

export function Layout({ children }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("auth_token");
    navigate("/login");
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 bg-card border-r border-border flex flex-col">
        <div className="p-6 border-b border-border">
          <h1 className="text-2xl font-bold text-primary">Humanizer Test Bench</h1>
          <p className="text-sm text-muted-foreground mt-1">Benchmark Analysis Platform</p>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )
              }
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-border">
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2 w-full rounded-md text-sm font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            <LogOut className="h-5 w-5" />
            Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto bg-background">
          <div className="container mx-auto p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
