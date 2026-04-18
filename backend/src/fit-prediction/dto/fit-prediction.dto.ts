import { IsString, IsNumber, IsOptional, IsArray, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';

export class MeasurementDto {
  @IsNumber()
  shoulderWidthCm: number;

  @IsNumber()
  chestWidthCm: number;

  @IsNumber()
  torsoLengthCm: number;

  @IsOptional()
  @IsNumber()
  waistCm?: number;

  @IsOptional()
  @IsNumber()
  hipCm?: number;

  @IsOptional()
  @IsNumber()
  inseamCm?: number;

  @IsOptional()
  @IsNumber()
  confidence?: number;

  @IsOptional()
  @IsArray()
  detectedRegions?: string[];
}

export class CreateFitPredictionDto {
  @IsString()
  sessionId: string;

  @IsOptional()
  @IsString()
  userId?: string;

  @IsString()
  productId: string;

  @ValidateNested()
  @Type(() => MeasurementDto)
  measurements: MeasurementDto;
}

export class FitResultDto {
  predictionId: string;
  decision: string;
  confidence: number;
  details: any;
  styleRecommendations: any[];
}
