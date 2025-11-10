import "./App.css";

import { useState } from "react";

import Header from "./components/Header";
import UploadButton from "./components/UploadButton";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResults] = useState({
    vit: { gradcam_image: "", predicted_class: "", confidence: 0 },
    convnext: { gradcam_image: "", predicted_class: "", confidence: 0 },
    coatnet: { gradcam_image: "", predicted_class: "", confidence: 0 },
  });
  const [loading, setLoading] = useState(false);

  // Submit for normalized
  // Submit for normalized
  const handleNormalized = async () => {
    if (!image) {
      alert("Please enter an image first.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", image);
      formData.append("normalize_stains", true);

      const res = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        alert("Failed to classify the SCC image. Try again!");
        return; // Add return to prevent further execution
      }

      const data = await res.json();

      // FIX: Use the correct keys from the response
      setResults({
        vit: {
          gradcam_image: data.models.vit.gradcam_image,
          predicted_class: data.models.vit.predicted_class,
          confidence: data.models.vit.confidence,
        },
        convnext: {
          gradcam_image: data.models.convnext.gradcam_image,
          predicted_class: data.models.convnext.predicted_class,
          confidence: data.models.convnext.confidence,
        },
        coatnet: {
          gradcam_image: data.models.coatnet.gradcam_image,
          predicted_class: data.models.coatnet.predicted_class,
          confidence: data.models.coatnet.confidence,
        },
      });
    } catch (err) {
      alert("Error: " + err); // Fixed string concatenation
    } finally {
      setLoading(false);
    }
  };

  // Classify without stain normalization
  const handleWithoutNormalized = async () => {
    if (!image) {
      alert("Please enter an image first.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", image);
      formData.append("normalize_stains", "false");

      const res = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        alert("Failed to classify the SCC image. Try again!");
        return; // Add return to prevent further execution
      }

      const data = await res.json();

      // FIX: Use the correct keys from the response
      setResults({
        vit: {
          gradcam_image: data.models.vit.gradcam_image,
          predicted_class: data.models.vit.predicted_class,
          confidence: data.models.vit.confidence,
        },
        convnext: {
          gradcam_image: data.models.convnext.gradcam_image,
          predicted_class: data.models.convnext.predicted_class,
          confidence: data.models.convnext.confidence,
        },
        coatnet: {
          gradcam_image: data.models.coatnet.gradcam_image,
          predicted_class: data.models.coatnet.predicted_class,
          confidence: data.models.coatnet.confidence,
        },
      });
    } catch (err) {
      alert("Error: " + err.message); // Fixed string concatenation
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header />

      {/* Main Content */}
      <main className="container-fluid bg-white px-2 py-4 px-md-3 py-md-5">
        <div className="container-md mx-md-auto p-0">
          {/* Functionalities */}
          <section className="container-fluid d-flex flex-wrap align-items-center justify-content-between gap-4 gap-lg-0">
            <div className="d-flex align-items-center gap-2">
              <p className="m-0 fw-semibold text-black">SCC File:</p>
              <div>
                <input
                  className="form-control"
                  type="file"
                  id="sccFile"
                  accept="image/*"
                  placeholder="SCC File Input"
                  disabled={loading}
                  onChange={(e) => setImage(e.target.files[0])}
                />
              </div>
            </div>

            <div className="d-flex gap-2">
              {/* Disabled during loading */}
              <div className="d-flex gap-2">
                <UploadButton onClick={handleNormalized} disable={loading}>
                  {loading ? "Processing..." : "w/Normalization"}
                </UploadButton>

                <UploadButton
                  onClick={handleWithoutNormalized}
                  disable={loading}
                >
                  {loading ? "Processing..." : "w/o Normalization"}
                </UploadButton>
              </div>
            </div>
          </section>

          {/* Output Images */}
          <section className="container-fluid mt-5">
            <div className="row row-cols-md-2 row-cols-xl-3 gap-5 gap-md-0">
              <div className="col">
                <div className="d-flex flex-column gap-3">
                  <p className="m-0 text-center fw-semibold">
                    Vit B-16 (Park et al.)
                  </p>
                  <img
                    src={
                      result.vit.gradcam_image || "https://placehold.co/600x600"
                    }
                    alt="Output Vit-B16"
                    className="rounded-2 shadow"
                  />
                  <div className="rounded-3 border border-2 border-black p-2 mt-3 d-flex flex-column gap-3 align-items-center">
                    <p className="fs-5 fw-semibold m-0">Results</p>
                    <div className="text-center">
                      <p className="m-0">
                        <span className="fw-medium">Prediction:</span>{" "}
                        {result.vit.predicted_class || "N/A"}
                      </p>
                      <p className="m-0">
                        <span className="fw-medium">Confidence:</span>{" "}
                        {result.vit.confidence || "N/A"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="col">
                <div className="d-flex flex-column gap-3">
                  <p className="m-0 text-center fw-semibold">ConvNeXt</p>
                  <img
                    src={
                      result.convnext.gradcam_image ||
                      "https://placehold.co/600x600"
                    }
                    alt="Output ConvNeXt"
                    className="rounded-2 shadow"
                  />
                  <div className="rounded-3 border border-2 border-black p-2 mt-3 d-flex flex-column gap-3 align-items-center">
                    <p className="fs-5 fw-semibold m-0">Results</p>
                    <div className="text-center">
                      <p className="m-0">
                        <span className="fw-medium">Prediction:</span>{" "}
                        {result.convnext.predicted_class || "N/A"}
                      </p>
                      <p className="m-0">
                        <span className="fw-medium">Confidence:</span>{" "}
                        {result.convnext.confidence || "N/A"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="col mt-md-4 mt-lg-0">
                <div className="d-flex flex-column gap-3">
                  <p className="m-0 text-center fw-semibold">CoAtNet</p>
                  <img
                    src={
                      result.coatnet.gradcam_image ||
                      "https://placehold.co/600x600"
                    }
                    alt="Output CoAtNet"
                    className="rounded-2 shadow"
                  />
                  <div className="rounded-3 border border-2 border-black p-2 mt-3 d-flex flex-column gap-3 align-items-center">
                    <p className="fs-5 fw-semibold m-0">Results</p>
                    <div className="text-center">
                      <p className="m-0">
                        <span className="fw-medium">Prediction:</span>{" "}
                        {result.coatnet.predicted_class || "N/A"}
                      </p>
                      <p className="m-0">
                        <span className="fw-medium">Confidence:</span>{" "}
                        {result.coatnet.confidence || "N/A"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

export default App;
