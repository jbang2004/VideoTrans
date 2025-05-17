import { useLanguage } from "@/hooks/use-language";
import Carousel from "@/components/carousel";
import GenerateSection from "@/components/generate-section";
import GallerySection from "@/components/gallery-section";
import { useCarouselData, useGalleryData } from "@/data/home-data";
import { BlurFade } from "@/components/magicui/blur-fade";

export default function Home() {
  const { t } = useLanguage();
  const carouselItems = useCarouselData();
  const galleryItems = useGalleryData();

  return (
    <section className="max-w-7xl mx-auto">
      <BlurFade delay={0.25} inView={true}>
        <div>
          <Carousel items={carouselItems} />
        </div>
      </BlurFade>
      <BlurFade delay={0.25 + 0.15} inView={true}>
        <div className="mt-12">
          <GenerateSection />
        </div>
      </BlurFade>
      <BlurFade delay={0.25 + 0.15 + 0.15} inView={true}>
        <div className="mt-12">
          <GallerySection items={galleryItems} />
        </div>
      </BlurFade>
    </section>
  );
}
