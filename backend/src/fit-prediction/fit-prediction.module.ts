import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { FitPredictionController } from './fit-prediction.controller';
import { FitPredictionService } from './fit-prediction.service';

@Module({
  imports: [HttpModule],
  controllers: [FitPredictionController],
  providers: [FitPredictionService],
  exports: [FitPredictionService]
})
export class FitPredictionModule {}
