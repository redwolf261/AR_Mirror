import {
  Body,
  Controller,
  Get,
  Param,
  Post,
  Query,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { MeasurementsService } from './measurements.service';
import { CreateMeasurementDto } from './dto/measurement.dto';

@Controller('measurements')
export class MeasurementsController {
  constructor(private readonly measurementsService: MeasurementsService) {}

  /**
   * POST /measurements
   * Called by the AR Mirror desktop app to upload a body measurement snapshot.
   */
  @Post()
  @HttpCode(HttpStatus.CREATED)
  async create(@Body() dto: CreateMeasurementDto) {
    return this.measurementsService.create(dto);
  }

  /**
   * GET /measurements/session/:sessionId
   * Returns all measurements for a specific AR session.
   */
  @Get('session/:sessionId')
  async findBySession(@Param('sessionId') sessionId: string) {
    return this.measurementsService.findBySession(sessionId);
  }

  /**
   * GET /measurements/recent?limit=50
   * Returns the most recent N measurements (for analytics).
   */
  @Get('recent')
  async findRecent(@Query('limit') limit?: string) {
    return this.measurementsService.findRecent(limit ? parseInt(limit, 10) : 50);
  }
}
