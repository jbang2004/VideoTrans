import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { CarouselItem } from "@shared/types";

interface CarouselProps {
  items: CarouselItem[];
  autoplayInterval?: number;
}

export default function Carousel({ items, autoplayInterval = 5000 }: CarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  const nextSlide = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % items.length);
  };

  const prevSlide = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + items.length) % items.length);
  };

  const goToSlide = (index: number) => {
    setCurrentIndex(index);
  };

  useEffect(() => {
    if (isPaused) return;

    const intervalId = setInterval(() => {
      nextSlide();
    }, autoplayInterval);

    return () => clearInterval(intervalId);
  }, [currentIndex, isPaused, autoplayInterval]);

  return (
    <div 
      className="relative mb-12 overflow-hidden rounded-2xl" 
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={currentIndex}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="carousel-item bg-card rounded-2xl overflow-hidden"
        >
          <div className="grid md:grid-cols-2 gap-0">
            {/* Left Image */}
            <div className="relative">
              <img 
                src={items[currentIndex].imageUrl} 
                alt={items[currentIndex].title} 
                className="object-cover w-full h-full" 
              />
              <div className="absolute inset-0 bg-gradient-to-r from-black/50 to-transparent pointer-events-none"></div>
            </div>
            
            {/* Right Content */}
            <div className="flex items-center p-8">
              <div>
                {items[currentIndex].tag && (
                  <span className="inline-block px-2 py-1 mb-2 text-xs font-medium bg-muted rounded-full">
                    {items[currentIndex].tag}
                  </span>
                )}
                <h2 className="text-2xl md:text-3xl font-bold mb-2">{items[currentIndex].title}</h2>
                <p className="text-muted-foreground mb-6">{items[currentIndex].description}</p>
                <a 
                  href={items[currentIndex].buttonLink}
                  className="bg-card border border-border rounded-full px-6 py-2 font-medium hover:bg-muted transition-colors inline-block"
                >
                  {items[currentIndex].buttonText}
                </a>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Carousel Indicators */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
        {items.map((_, index) => (
          <button 
            key={index}
            className={cn(
              "carousel-indicator h-2 rounded-full",
              index === currentIndex 
                ? "active w-6 bg-primary" 
                : "w-2 bg-primary/40"
            )}
            onClick={() => goToSlide(index)}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
      
      {/* Carousel Navigation */}
      <button 
        className="absolute left-4 top-1/2 transform -translate-y-1/2 w-10 h-10 bg-background/80 border border-border rounded-full flex items-center justify-center hover:bg-background transition-colors"
        onClick={prevSlide}
        aria-label="Previous slide"
      >
        <ChevronLeft className="h-6 w-6" />
      </button>
      <button 
        className="absolute right-4 top-1/2 transform -translate-y-1/2 w-10 h-10 bg-background/80 border border-border rounded-full flex items-center justify-center hover:bg-background transition-colors"
        onClick={nextSlide}
        aria-label="Next slide"
      >
        <ChevronRight className="h-6 w-6" />
      </button>
    </div>
  );
}
