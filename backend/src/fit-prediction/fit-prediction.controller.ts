import { Controller, Post, Get, Body, Param, Query, UseGuards } from '@nestjs/common';
import { FitPredictionService } from './fit-prediction.service';
import { CreateFitPredictionDto, FitResultDto } from './dto/fit-prediction.dto';

@Controller('fit-prediction')
export class FitPredictionController {
  constructor(private readonly fitService: FitPredictionService) {}

  @Post()
  async predictFit(@Body() dto: CreateFitPredictionDto): Promise<FitResultDto> {
    return this.fitService.predictFit(dto);
  }

  @Get('product/:productId')
  async getProductFitHistory(@Param('productId') productId: string) {
    return this.fitService.getProductFitHistory(productId);
  }

  @Post('feedback/:predictionId')
  async submitFeedback(
    @Param('predictionId') predictionId: string,
    @Body() feedback: { actualFit: string; wasReturned: boolean; userFeedback?: string }
  ) {
    return this.fitService.submitFeedback(predictionId, feedback);
  }
}
