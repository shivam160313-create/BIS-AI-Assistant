import { useState } from "react";

const features = [
  {
    id: "assistant",
    icon: "✦",
    title: "AI Assistant",
    description: "Ask anything about BIS standards and services.",
  },
  {
    id: "standards",
    icon: "🔎",
    title: "Standards Search",
    description: "Find relevant Indian Standards.",
  },
  {
    id: "document",
    icon: "📄",
    title: "Document Q&A",
    description: "Ask questions from BIS documents.",
  },
  {
    id: "certification",
    icon: "📜",
    title: "Certification Guide",
    description: "Understand BIS licensing and certification.",
  },
  {
    id: "compliance",
    icon: "🛡️",
    title: "Compliance",
    description: "Understand BIS compliance requirements.",
  },
];

const suggestions = {
  assistant: [
    "What is ISI Mark?",
    "What is BIS?",
    "Is BIS certification compulsory?",
  ],
  standards: [
    "Cement",
    "Electric fan",
    "Helmet",
    "Water bottle",
  ],
  document: [
    "What are the BIS licence requirements?",
    "What is the certification process?",
    "What is Scheme-I?",
  ],
  certification: [
    "How can I get a BIS licence?",
    "What documents are required?",
    "How does BIS inspection work?",
  ],
  compliance: [
    "What are BIS compliance requirements?",
    "What is a QCO?",
    "What testing is required?",
  ],
};

function App() {
  const [activeFeature, setActiveFeature] = useState("assistant");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  const currentFeature = features.find(
    (feature) => feature.id === activeFeature
  );

  const sendRequest = async (text = question) => {
    if (!text.trim()) return;

    setQuestion(text);
    setLoading(true);
    setAnswer("");
    setSources([]);

    let endpoint = "ask";
    let body = { question: text };

    if (activeFeature === "standards") {
      endpoint = "standards";
      body = { product: text };
    }

    if (activeFeature === "document") {
      endpoint = "document-qa";
    }

    if (activeFeature === "certification") {
      endpoint = "certification";
    }

    if (activeFeature === "compliance") {
      endpoint = "compliance";
    }

    try {
      const response = await fetch(
        `/api/${endpoint}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
        }
      );

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const data = await response.json();

      setAnswer(data.answer || "No answer received.");
      setSources(data.sources || []);
    } catch (error) {
      setAnswer(
        "Unable to connect to BIS AI. Please make sure the backend server is running."
      );
    }

    setLoading(false);
  };

  const uploadDocument = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadMessage("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        "/api/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      setUploadMessage(
        `✅ ${data.filename} uploaded successfully`
      );

      setSelectedFile(null);
    } catch (error) {
      setUploadMessage(
        "❌ Document upload failed. Please try again."
      );
    }

    setUploading(false);
  };

  const selectFeature = (id) => {
    setActiveFeature(id);
    setAnswer("");
    setSources([]);
    setUploadMessage("");

    if (id === "assistant" || id === "standards" || id === "document") {
      setQuestion("");
    }

    if (id === "certification") {
      setQuestion(
        "What is the complete process for getting a BIS licence?"
      );
    }

    if (id === "compliance") {
      setQuestion(
        "What are the requirements for BIS product compliance?"
      );
    }
  };

  const handleSubmit = () => {
    sendRequest();
  };

  return (
    <div className="app">

      <div className="glow glow1"></div>
      <div className="glow glow2"></div>

      <nav className="navbar">

        <div className="brand">

          <div className="logo">
            🇮🇳
          </div>

          <div>
            <div className="brand-title">
              BIS AI
            </div>

            <div className="brand-subtitle">
              Standards Intelligence Platform
            </div>
          </div>

        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          AI System Online
        </div>

      </nav>

      <div className="feature-nav">

        {features.map((feature) => (
          <button
            key={feature.id}
            className={
              activeFeature === feature.id
                ? "active"
                : ""
            }
            onClick={() => selectFeature(feature.id)}
          >
            <span>{feature.icon}</span>
            {feature.title}
          </button>
        ))}

      </div>

      <main className="container">

        <section className="hero">

          <div className="badge">
            ✦ AI-Powered BIS Knowledge Assistant
          </div>

          <h1>
            Your Intelligent Guide to
            <span>Indian Standards</span>
          </h1>

          <p>
            Ask questions about BIS standards, certification,
            compliance and official documents — powered by AI.
          </p>

        </section>

        <section className="search-card">

          <div className="search-header">

            <div>

              <div className="feature-label">
                {currentFeature.icon} {currentFeature.title}
              </div>

              <h2>
                {activeFeature === "standards"
                  ? "Find your Indian Standard"
                  : activeFeature === "document"
                  ? "Ask questions from BIS documents"
                  : activeFeature === "certification"
                  ? "Get BIS Certification Guidance"
                  : activeFeature === "compliance"
                  ? "Check BIS Compliance"
                  : "How can I help you?"}
              </h2>

              <p>
                {activeFeature === "standards"
                  ? "Enter your product or product category."
                  : activeFeature === "document"
                  ? "Ask questions using information from the BIS knowledge base."
                  : activeFeature === "certification"
                  ? "Understand the BIS licence and certification process."
                  : activeFeature === "compliance"
                  ? "Understand product compliance requirements."
                  : "Ask anything related to Indian Standards and BIS."}
              </p>

            </div>

            <div className="ai-icon">
              ✦
            </div>

          </div>

          {activeFeature === "document" && (

            <div className="upload-box">

              <div className="upload-icon">
                📄
              </div>

              <div className="upload-content">

                <strong>
                  Upload BIS Document
                </strong>

                <small>
                  PDF or TXT document
                </small>

                <input
                  type="file"
                  accept=".pdf,.txt"
                  onChange={(e) =>
                    setSelectedFile(e.target.files[0])
                  }
                />

              </div>

              <button
                className="upload-button"
                onClick={uploadDocument}
                disabled={
                  uploading ||
                  !selectedFile
                }
              >
                {uploading
                  ? "Uploading..."
                  : "Upload"}
              </button>

            </div>

          )}

          {uploadMessage && (

            <div className="upload-message">
              {uploadMessage}
            </div>

          )}

          <div className="input-wrapper">

            <textarea
              value={question}
              onChange={(e) =>
                setQuestion(e.target.value)
              }
              onKeyDown={(e) => {

                if (
                  e.key === "Enter" &&
                  !e.shiftKey
                ) {
                  e.preventDefault();
                  handleSubmit();
                }

              }}
              placeholder={
                activeFeature === "standards"
                  ? "Example: Cement, helmet, electric fan..."
                  : activeFeature === "document"
                  ? "Example: What are the licence requirements?"
                  : activeFeature === "certification"
                  ? "Example: How can I get a BIS licence?"
                  : activeFeature === "compliance"
                  ? "Example: What are the BIS compliance requirements?"
                  : "Example: What is ISI Mark?"
              }
            />

            <button
              onClick={handleSubmit}
              disabled={
                loading ||
                !question.trim()
              }
            >
              {loading
                ? "Processing..."
                : activeFeature === "standards"
                ? "Search Standards →"
                : activeFeature === "document"
                ? "Ask Document →"
                : activeFeature === "certification"
                ? "Get Guidance →"
                : activeFeature === "compliance"
                ? "Check Compliance →"
                : "Ask BIS AI →"}
            </button>

          </div>

          <div className="suggestions">

            <span>
              Try asking:
            </span>

            {suggestions[activeFeature].map(
              (item) => (

                <button
                  key={item}
                  onClick={() =>
                    sendRequest(item)
                  }
                >
                  {item}
                </button>

              )
            )}

          </div>

        </section>

        {loading && (

          <div className="loading-card">

            <div className="loader"></div>

            <div>

              <strong>
                {activeFeature === "standards"
                  ? "Searching BIS standards..."
                  : activeFeature === "document"
                  ? "Searching BIS documents..."
                  : activeFeature === "certification"
                  ? "Analyzing certification requirements..."
                  : activeFeature === "compliance"
                  ? "Checking compliance information..."
                  : "Analyzing BIS knowledge..."}
              </strong>

              <p>
                Finding relevant BIS information
              </p>

            </div>

          </div>

        )}

        {answer && !loading && (

          <section className="answer-card">

            <div className="answer-heading">

              <div className="answer-icon">
                ✦
              </div>

              <div>

                <h2>
                  {activeFeature === "standards"
                    ? "Standards Search Result"
                    : activeFeature === "document"
                    ? "Document Answer"
                    : activeFeature === "certification"
                    ? "Certification Guidance"
                    : activeFeature === "compliance"
                    ? "Compliance Analysis"
                    : "AI Answer"}
                </h2>

                <p>
                  Generated using BIS knowledge sources
                </p>

              </div>

            </div>

            <div className="answer-content">
              {answer}
            </div>

            {sources.length > 0 && (

              <div className="sources">

                <div className="sources-title">

                  <span>
                    📚
                  </span>

                  <div>

                    <strong>
                      Knowledge Sources
                    </strong>

                    <small>
                      BIS documents used
                    </small>

                  </div>

                </div>

                <div className="source-list">

                  {sources.map(
                    (source, index) => (

                      <div
                        className="source"
                        key={source}
                      >

                        <div className="source-number">
                          {index + 1}
                        </div>

                        <div>

                          <strong>
                            {source}
                          </strong>

                          <small>
                            BIS Knowledge Base
                          </small>

                        </div>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}

          </section>

        )}

        <section className="features">

          {features.map((feature) => (

            <div
              className={`feature ${
                activeFeature === feature.id
                  ? "selected"
                  : ""
              }`}
              key={feature.id}
              onClick={() =>
                selectFeature(feature.id)
              }
            >

              <div className="feature-icon">
                {feature.icon}
              </div>

              <h3>
                {feature.title}
              </h3>

              <p>
                {feature.description}
              </p>

            </div>

          ))}

        </section>

        <footer>

          <span>
            BIS AI Assistant
          </span>

          <span>•</span>

          <span>
            Hackathon Prototype
          </span>

          <span>•</span>

          <span>
            RAG + AI
          </span>

        </footer>

      </main>

    </div>
  );
}

export default App;