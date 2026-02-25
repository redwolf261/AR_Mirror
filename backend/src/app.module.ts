import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PrismaModule } from './prisma/prisma.module';
import { AuthModule } from './auth/auth.module';
import { ProductsModule } from './products/products.module';
import { MeasurementsModule } from './measurements/measurements.module';
import { FitPredictionModule } from './fit-prediction/fit-prediction.module';
import { OrdersModule } from './orders/orders.module';
import { AnalyticsModule } from './analytics/analytics.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true
    }),
    PrismaModule,
    AuthModule,
    ProductsModule,
    MeasurementsModule,
    FitPredictionModule,
    OrdersModule,
    AnalyticsModule
  ]
})
export class AppModule {}
