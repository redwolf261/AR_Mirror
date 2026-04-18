import {
  IsNumber,
  IsOptional,
  IsString,
  IsArray,
  Min,
  Max,
} from 'class-validator';

export class CreateMeasurementDto {
  @IsNumber()
  @Min(0)
  shoulderWidthCm: number;

  @IsNumber()
  @Min(0)
  chestWidthCm: number;

  @IsNumber()
  @Min(0)
  torsoLengthCm: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  waistCm?: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  hipCm?: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  inseamCm?: number;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(1)
  confidence?: number = 1.0;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  detectedRegions?: string[] = ['upper_body'];

  @IsOptional()
  @IsString()
  garmentSku?: string;

  @IsOptional()
  @IsString()
  bodyShape?: string;

  @IsOptional()
  @IsString()
  sessionId?: string;
}
