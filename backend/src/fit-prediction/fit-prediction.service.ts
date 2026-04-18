import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { CreateFitPredictionDto, FitResultDto } from './dto/fit-prediction.dto';

@Injectable()
export class FitPredictionService {
  private pythonServiceUrl: string;

  constructor(
    private prisma: PrismaService,
    private httpService: HttpService,
    private config: ConfigService
  ) {
    this.pythonServiceUrl = this.config.get('PYTHON_SERVICE_URL') || 'http://localhost:8000';
  }

  async predictFit(dto: CreateFitPredictionDto): Promise<FitResultDto> {
    // 1. Get product and garment specs
    const product = await this.prisma.product.findUnique({
      where: { id: dto.productId },
      include: { garmentSpecs: true, skuCorrections: { where: { isActive: true } } }
    });

    if (!product || !product.garmentSpecs) {
      throw new HttpException('Product not found or specs missing', HttpStatus.NOT_FOUND);
    }

    // 2. Call Python ML service for fit prediction
    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.pythonServiceUrl}/predict-fit`, {
          measurements: {
            shoulder_width_cm: dto.measurements.shoulderWidthCm,
            chest_width_cm: dto.measurements.chestWidthCm,
            torso_length_cm: dto.measurements.torsoLengthCm,
            waist_cm: dto.measurements.waistCm,
            hip_cm: dto.measurements.hipCm,
            inseam_cm: dto.measurements.inseamCm
          },
          garment_specs: {
            shoulder_width_cm: product.garmentSpecs.shoulderWidthCm,
            chest_width_cm: product.garmentSpecs.chestWidthCm,
            length_cm: product.garmentSpecs.lengthCm,
            waist_cm: product.garmentSpecs.waistCm,
            hip_cm: product.garmentSpecs.hipCm,
            inseam_cm: product.garmentSpecs.inseamCm,
            tight_tolerance: product.garmentSpecs.tightTolerance,
            loose_tolerance: product.garmentSpecs.looseTolerance
          },
          sku_correction: product.skuCorrections[0] || null,
          product_category: product.category
        })
      );

      const fitResult = response.data;

      // 3. Store measurement
      const measurement = await this.prisma.measurement.create({
        data: {
          sessionId: dto.sessionId,
          userId: dto.userId,
          shoulderWidthCm: dto.measurements.shoulderWidthCm,
          chestWidthCm: dto.measurements.chestWidthCm,
          torsoLengthCm: dto.measurements.torsoLengthCm,
          waistCm: dto.measurements.waistCm,
          hipCm: dto.measurements.hipCm,
          inseamCm: dto.measurements.inseamCm,
          confidence: dto.measurements.confidence || 0.8,
          detectedRegions: JSON.stringify(dto.measurements.detectedRegions || [])
        }
      });

      // 4. Store fit prediction
      const prediction = await this.prisma.fitPrediction.create({
        data: {
          measurementId: measurement.id,
          productId: product.id,
          decision: fitResult.decision,
          confidence: fitResult.confidence,
          shoulderFit: fitResult.details?.shoulder,
          chestFit: fitResult.details?.chest,
          waistFit: fitResult.details?.waist,
          hipFit: fitResult.details?.hip,
          correctionApplied: !!product.skuCorrections[0],
          correctionId: product.skuCorrections[0]?.id
        }
      });

      return {
        predictionId: prediction.id,
        decision: fitResult.decision,
        confidence: fitResult.confidence,
        details: fitResult.details,
        styleRecommendations: fitResult.style_recommendations || []
      };

    } catch (error) {
      console.error('Python service error:', error);
      throw new HttpException('Fit prediction service unavailable', HttpStatus.SERVICE_UNAVAILABLE);
    }
  }

  async getProductFitHistory(productId: string) {
    const predictions = await this.prisma.fitPrediction.findMany({
      where: { productId },
      include: {
        measurement: true
      },
      orderBy: { timestamp: 'desc' },
      take: 100
    });

    return {
      productId,
      totalPredictions: predictions.length,
      fitDistribution: {
        tight: predictions.filter(p => p.decision === 'TIGHT').length,
        good: predictions.filter(p => p.decision === 'GOOD').length,
        loose: predictions.filter(p => p.decision === 'LOOSE').length
      },
      predictions: predictions.slice(0, 20) // Return latest 20
    };
  }

  async submitFeedback(predictionId: string, feedback: any) {
    return this.prisma.fitPrediction.update({
      where: { id: predictionId },
      data: {
        actualFit: feedback.actualFit,
        wasReturned: feedback.wasReturned,
        userFeedback: feedback.userFeedback
      }
    });
  }
}
