import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateProductDto } from './dto/product.dto';

@Injectable()
export class ProductsService {
  constructor(private prisma: PrismaService) {}

  async findAll(query: { category?: string; brand?: string; limit?: number }) {
    return this.prisma.product.findMany({
      where: {
        isActive: true,
        ...(query.category && { category: query.category }),
        ...(query.brand && { brand: query.brand })
      },
      include: { garmentSpecs: true },
      take: query.limit || 50,
      orderBy: { createdAt: 'desc' }
    });
  }

  async findOne(id: string) {
    return this.prisma.product.findUnique({
      where: { id },
      include: { 
        garmentSpecs: true,
        skuCorrections: { where: { isActive: true } }
      }
    });
  }

  async create(dto: CreateProductDto) {
    return this.prisma.product.create({
      data: {
        sku: dto.sku,
        name: dto.name,
        category: dto.category,
        subcategory: dto.subcategory,
        brand: dto.brand,
        price: dto.price,
        currency: dto.currency || 'INR',
        images: dto.images || [],
        description: dto.description,
        garmentSpecs: dto.garmentSpecs ? {
          create: dto.garmentSpecs
        } : undefined
      },
      include: { garmentSpecs: true }
    });
  }
}
