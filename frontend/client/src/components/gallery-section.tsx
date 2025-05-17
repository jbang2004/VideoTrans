import { useState } from "react";
import { Link } from "wouter";
import { ChevronRight } from "lucide-react";
import { motion } from "framer-motion";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";

interface GalleryItem {
  id: string;
  title: string;
  image: string;
  authorName?: string;
  authorAvatar?: string;
  link: string;
}

interface GallerySectionProps {
  items: GalleryItem[];
}

export default function GallerySection({ items }: GallerySectionProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const { language } = useLanguage();
  const { theme } = useTheme();

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">{language === "zh" ? "画廊" : "Gallery"}</h2>
        <Link 
          href="/feed" 
          className="text-sm text-primary flex items-center"
        >
          {language === "zh" ? "打开画廊" : "Open Gallery"}
          <ChevronRight className="h-4 w-4 ml-1" />
        </Link>
      </div>
      
      <div className="grid grid-cols-1 xs:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3 sm:gap-4">
        {items.map((item, index) => (
          <Link 
            key={item.id} 
            href={item.link}
            className="block"
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <div className="relative rounded-xl overflow-hidden" style={{ aspectRatio: '1/1' }}>
              <motion.img 
                src={item.image} 
                alt={item.title} 
                className="object-cover w-full h-full"
                animate={{ 
                  scale: hoveredIndex === index ? 1.05 : 1,
                  transition: { duration: 0.3 }
                }}
              />
              
              {/* Hover overlay with title */}
              <motion.div 
                className="absolute inset-0 bg-black/40 flex flex-col justify-end p-3"
                initial={{ opacity: 0 }}
                animate={{ 
                  opacity: hoveredIndex === index ? 1 : 0,
                  transition: { duration: 0.3 }
                }}
              >
                <h3 className="text-white font-medium text-sm">{item.title}</h3>
                
                {item.authorName && (
                  <div className="flex items-center mt-1">
                    {item.authorAvatar && (
                      <img 
                        src={item.authorAvatar} 
                        alt={item.authorName} 
                        className="w-5 h-5 rounded-full mr-1"
                      />
                    )}
                    <span className="text-white/80 text-xs">{item.authorName}</span>
                  </div>
                )}
              </motion.div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
