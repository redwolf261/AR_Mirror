import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable CORS for mobile app
  app.enableCors({
    origin: ['http://localhost:19006', 'exp://localhost:19000'], // Expo dev
    credentials: true
  });
  
  // Global validation pipe
  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    transform: true
  }));
  
  const port = process.env.PORT || 3000;
  await app.listen(port);
  
  console.log('='.repeat(60));
  console.log('CHIC INDIA AR PLATFORM - BACKEND API');
  console.log('='.repeat(60));
  console.log(`✓ Server running on http://localhost:${port}`);
  console.log(`✓ Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`✓ Database: PostgreSQL`);
  console.log(`✓ Python ML Service: ${process.env.PYTHON_SERVICE_URL}`);
  console.log('='.repeat(60));
}

bootstrap();
