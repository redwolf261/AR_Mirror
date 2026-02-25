/**
 * Home Screen — Garment catalogue browser
 * Fetches products from the NestJS backend and lets the user pick one to try on.
 */
import React, { useEffect, useState, useCallback } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { router } from 'expo-router';
import { useARStore } from '../src/store/arStore';
import { apiClient } from '../src/services/api';

interface Product {
  id: string;
  sku: string;
  name: string;
  brand: string;
  price: number;
  currency: string;
  images: string[];
  category: string;
}

export default function HomeScreen() {
  const [products, setProducts]   = useState<Product[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const setSelectedProduct        = useARStore(s => s.setSelectedProduct);

  const loadProducts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await apiClient.get<Product[]>('/products');
      setProducts(data);
    } catch (e: any) {
      setError('Could not load garments. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadProducts(); }, [loadProducts]);

  const onPressProduct = (product: Product) => {
    setSelectedProduct(product.id);
    router.push('/tryon');
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00d4ff" />
        <Text style={styles.hint}>Loading garments…</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
        <Pressable style={styles.retryBtn} onPress={loadProducts}>
          <Text style={styles.retryText}>Retry</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={products}
        keyExtractor={item => item.id}
        numColumns={2}
        contentContainerStyle={styles.grid}
        renderItem={({ item }) => (
          <Pressable style={styles.card} onPress={() => onPressProduct(item)}>
            {item.images?.[0] ? (
              <Image source={{ uri: item.images[0] }} style={styles.thumb} />
            ) : (
              <View style={[styles.thumb, styles.thumbPlaceholder]} />
            )}
            <Text style={styles.cardName} numberOfLines={2}>{item.name}</Text>
            <Text style={styles.cardBrand}>{item.brand}</Text>
            <Text style={styles.cardPrice}>
              {item.currency} {item.price.toFixed(0)}
            </Text>
          </Pressable>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container:       { flex: 1, backgroundColor: '#0a0a0a' },
  center:          { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 12 },
  hint:            { color: '#aaa', marginTop: 8 },
  errorText:       { color: '#ff6060', textAlign: 'center', paddingHorizontal: 24 },
  retryBtn:        { marginTop: 12, paddingHorizontal: 24, paddingVertical: 10, backgroundColor: '#1e3a5f', borderRadius: 8 },
  retryText:       { color: '#00d4ff', fontWeight: '600' },
  grid:            { padding: 8, gap: 8 },
  card:            { flex: 1, margin: 4, backgroundColor: '#1a1a1a', borderRadius: 12, overflow: 'hidden' },
  thumb:           { width: '100%', aspectRatio: 3 / 4, resizeMode: 'cover' } as any,
  thumbPlaceholder:{ backgroundColor: '#2a2a2a' },
  cardName:        { color: '#fff', fontSize: 13, fontWeight: '600', paddingHorizontal: 8, paddingTop: 6 },
  cardBrand:       { color: '#888', fontSize: 11, paddingHorizontal: 8 },
  cardPrice:       { color: '#00d4ff', fontSize: 13, fontWeight: '700', padding: 8 },
});
