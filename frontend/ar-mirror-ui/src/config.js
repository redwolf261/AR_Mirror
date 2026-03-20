const config = {
  API_BASE_URL: process.env.NODE_ENV === 'production'
    ? process.env.REACT_APP_API_URL || 'https://ar-mirror-backend.railway.app'
    : 'http://localhost:5050'
};

export default config;