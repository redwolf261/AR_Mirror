import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateMeasurementDto } from './dto/measurement.dto';

@Injectable()
export class MeasurementsService {
  private readonly logger = new Logger(MeasurementsService.name);

  constructor(private readonly prisma: PrismaService) {}

  /**
   * Create or reuse an ARSession and insert a Measurement row.
   * Returns the created measurement.
   */
  async create(dto: CreateMeasurementDto) {
    // Upsert session — if sessionId not supplied, create a fresh one
    let session = dto.sessionId
      ? await this.prisma.aRSession.findUnique({ where: { id: dto.sessionId } })
      : null;

    if (!session) {
      session = await this.prisma.aRSession.create({
        data: {
          deviceInfo: { source: 'ar-mirror', garmentSku: dto.garmentSku ?? null },
        },
      });
    }

    const measurement = await this.prisma.measurement.create({
      data: {
        sessionId:      session.id,
        shoulderWidthCm: dto.shoulderWidthCm,
        chestWidthCm:    dto.chestWidthCm,
        torsoLengthCm:   dto.torsoLengthCm,
        waistCm:         dto.waistCm,
        hipCm:           dto.hipCm,
        inseamCm:        dto.inseamCm,
        confidence:      dto.confidence ?? 1.0,
        detectedRegions: dto.detectedRegions ?? ['upper_body'],
      },
    });

    this.logger.log(
      `Measurement saved  session=${session.id}  ` +
      `shoulder=${dto.shoulderWidthCm}cm  sku=${dto.garmentSku ?? '-'}  shape=${dto.bodyShape ?? '-'}`,
    );
    return { measurement, sessionId: session.id };
  }

  /** Retrieve all measurements for a given session */
  async findBySession(sessionId: string) {
    return this.prisma.measurement.findMany({
      where: { sessionId },
      orderBy: { timestamp: 'asc' },
    });
  }

  /** Retrieve recent measurements (latest N) */
  async findRecent(limit = 50) {
    return this.prisma.measurement.findMany({
      orderBy: { timestamp: 'desc' },
      take: limit,
    });
  }
}
