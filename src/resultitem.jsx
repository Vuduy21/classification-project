import React from 'react';

const getSentimentStyle = (sentiment) => {
  return sentiment
    ? { color: 'var(--bs-success)', icon: 'bi-emoji-smile-fill' }
    : { color: 'var(--bs-danger)', icon: 'bi-emoji-frown-fill' };
};

const ResultItem = ({ result, index }) => {
  const { review, predictions } = result;
  
  return (
    <div className="card mb-3 result-card">
      <div className="card-body">
        <h5 className="card-title mb-3">
          <i className="bi bi-chat-right-quote"></i> Review {index + 1}: "{review}"
        </h5>
        <ul className="list-group list-group-flush">
          {Object.entries(predictions).map(([model, pred]) => {
            const { color, icon } = getSentimentStyle(pred);
            return (
              <li key={model} className="list-group-item d-flex justify-content-between align-items-center">
                <strong>{model}</strong>: 
                <span style={{ color, fontWeight: 'bold' }}>
                  {pred ? 'Tích cực ' : 'Tiêu cực '}
                  <i className={`bi ${icon}`}></i>
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
};

export default ResultItem;