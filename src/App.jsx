import { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import './App.css';
import ResultItem from './resultitem';

function App() {
  const [reviews, setReviews] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModels, setSelectedModels] = useState([]);

  const models = [
    "logistic",
    "naive_bayes",
    "random_forest",
    "svm",
    "lstm",
    "bilstm",
    "distilbert",
    "deberta"
  ];

  const getPredictions = async (textsToPredict) => {
    setIsLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          texts: textsToPredict,
          models: selectedModels.length > 0 ? selectedModels : undefined
        }),
      });

      if (!response.ok) {
        throw new Error(`Lỗi từ server: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        const formattedResults = textsToPredict.map((review, index) => ({
          review: review,
          predictions: data.results[index],
        }));
        setResults(formattedResults);
      }
    } catch (err) {
      setError('Không thể kết nối đến backend. Vui lòng thử lại. ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = () => {
    const textList = reviews.split('\n').filter(text => text.trim() !== '');
    if (textList.length === 0) {
      setError('Vui lòng nhập ít nhất một review.');
      return;
    }
    getPredictions(textList);
  };

  const toggleModel = (model) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };

  const selectAll = () => {
    setSelectedModels(models);
  };

  return (
    <div className="App">
      <div className="content-wrapper">
        <div className="text-center mb-4 header">
          <h1 className="display-5">Phân loại Cảm xúc</h1>
          <p className="lead text-muted">Nhập review của khách hàng để phân tích.</p>
        </div>

        <div className="mb-3 input-section">
          <textarea
            className="form-control"
            rows="5"
            value={reviews}
            onChange={(e) => setReviews(e.target.value)}
            placeholder="Nhập các review của bạn (mỗi review một dòng)..."
            disabled={isLoading}
          />
        </div>

        <div className="mb-3 model-selection">
          <h4 className="mb-3">Chọn mô hình:</h4>
          <div className="form-check mb-2">
            <input
              type="checkbox"
              className="form-check-input"
              id="selectAll"
              checked={selectedModels.length === models.length}
              onChange={selectAll}
            />
            <label className="form-check-label" htmlFor="selectAll">
              Sử dụng tất cả
            </label>
          </div>
          {models.map(model => (
            <div key={model} className="form-check mb-2">
              <input
                type="checkbox"
                className="form-check-input"
                id={model}
                checked={selectedModels.includes(model)}
                onChange={() => toggleModel(model)}
              />
              <label className="form-check-label" htmlFor={model}>
                {model}
              </label>
            </div>
          ))}
        </div>

        <div className="d-flex justify-content-center gap-3 mb-3 button-section">
          <button
            className="btn btn-primary btn-lg"
            onClick={handlePredict}
            disabled={isLoading || !reviews.trim()}
          >
            {isLoading ? (
              <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            ) : (
              <i className="bi bi-robot"></i>
            )}
            {' '}Phân tích
          </button>
        </div>

        {error && <div className="alert alert-danger mt-3" role="alert">{error}</div>}

        {results.length > 0 && !isLoading && (
          <div className="mt-5 results-section">
            <h3 className="text-center mb-4">Kết quả phân tích</h3>
            {results.map((result, index) => (
              <ResultItem key={index} result={result} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

