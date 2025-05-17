// Common types used across the application

// Plan types
export interface Plan {
  id: string;
  name: string;
  monthlyPrice: number;
  yearlyPrice: number;
  features: string[];
  computeUnits: number;
  included: {
    fluxImages: number;
    realtimeImages: number;
    enhancedImages: number;
    trainingJobs: number;
    commercialLicense: boolean;
    priorityQueue: boolean;
    concurrentFluxGenerations: number;
    concurrentVideoGenerations: number;
    concurrentTrainings: number;
  };
}

// Carousel item types
export interface CarouselItem {
  id: string;
  title: string;
  description: string;
  buttonText: string;
  buttonLink: string;
  imageUrl: string;
  tag?: string;
}

// Generate card types
export interface GenerateCard {
  id: string;
  title: string;
  description: string;
  icon: string;
  link: string;
  isNew?: boolean;
  bgColor?: string;
  textColor?: string;
}

// Video model types
export interface VideoModel {
  id: string;
  name: string;
  description: string;
  duration: string;
  features: string[];
  isExpensiveModel?: boolean;
  icon: string;
}

// Gallery item types
export interface GalleryItem {
  id: string;
  title: string;
  image: string;
  authorName?: string;
  authorAvatar?: string;
  created: Date;
  link: string;
}

// FAQ item types
export interface FaqItem {
  question: string;
  answer: string;
}
