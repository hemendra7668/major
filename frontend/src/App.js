import "./App.css";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Index from "./components";

function App() {
  return (
    <Router>
      <div className="app">
        <header className="top">Patient Vital Sign Prediction and Analysis App</header>

        <Routes>
          <Route
            path="/"
            element={
              <main className="hero">
                <h1>AI Powered Healthcare Monitoring System</h1>
                <p>
                  Robo-Doc predicts and monitors vital health parameters using machine learning
                  algorithms, helping doctors make faster and smarter decisions.
                </p>

                {/* React Router Link instead of <a> */}
                <Link to="/index" className="btn">
                  Launch Application
                </Link>
              </main>
            }
          />

          <Route path="/Index" element={<Index />} />
        </Routes>

        <footer>© 2025 Robo-Doc | AI Healthcare Assistant</footer>
      </div>
    </Router>
  );
}

export default App;
