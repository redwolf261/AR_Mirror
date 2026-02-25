import { Stack } from 'expo-router';
import { StatusBar } from 'react-native';

export default function RootLayout() {
  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#0a0a0a" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#1a1a1a' },
          headerTintColor: '#ffffff',
          headerTitleStyle: { fontWeight: 'bold' },
          contentStyle: { backgroundColor: '#0a0a0a' },
        }}
      >
        <Stack.Screen name="index"   options={{ title: 'Chic India AR' }} />
        <Stack.Screen name="tryon"   options={{ title: 'Try On', headerShown: false }} />
        <Stack.Screen name="profile" options={{ title: 'My Measurements' }} />
      </Stack>
    </>
  );
}
