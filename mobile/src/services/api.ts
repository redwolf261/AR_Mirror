/**
 * Chic India Mobile - API Client
 * Communicates with NestJS backend
 */
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = __DEV__ 
  ? 'http://localhost:3000' 
  : 'https://api.chicindia.com';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add auth token to requests
apiClient.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem('authToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface Measurement {
  shoulderWidthCm: number;
  chestWidthCm: number;
  torsoLengthCm: number;
  waistCm?: number;
  hipCm?: number;
  inseamCm?: number;
  confidence?: number;
  detectedRegions?: string[];
}

export interface FitPredictionRequest {
  sessionId: string;
  userId?: string;
  productId: string;
  measurements: Measurement;
}

export interface FitPredictionResponse {
  predictionId: string;
  decision: 'TIGHT' | 'GOOD' | 'LOOSE';
  confidence: number;
  details: Record<string, string>;
  styleRecommendations: Array<{
    category: string;
    subcategory: string;
    reason: string;
    color: string;
    confidence: number;
  }>;
}

export const api = {
  // Fit prediction
  async predictFit(data: FitPredictionRequest): Promise<FitPredictionResponse> {
    const response = await apiClient.post('/fit-prediction', data);
    return response.data;
  },

  // Products
  async getProducts(filters?: { category?: string; brand?: string }) {
    const response = await apiClient.get('/products', { params: filters });
    return response.data;
  },

  async getProduct(id: string) {
    const response = await apiClient.get(`/products/${id}`);
    return response.data;
  },

  // Session management
  async createSession(deviceInfo: any) {
    const response = await apiClient.post('/sessions', { deviceInfo });
    return response.data;
  },

  // Feedback
  async submitFitFeedback(predictionId: string, feedback: {
    actualFit: string;
    wasReturned: boolean;
    userFeedback?: string;
  }) {
    const response = await apiClient.post(`/fit-prediction/feedback/${predictionId}`, feedback);
    return response.data;
  }
};

export default api;
