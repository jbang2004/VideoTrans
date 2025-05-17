import { Link } from "wouter";
import { ChevronRight } from "lucide-react";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { useGenerateCardData } from "@/data/home-data";

interface GenerateCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  link: string;
  isNew?: boolean;
}

function GenerateCard({ title, description, icon, link, isNew }: GenerateCardProps) {
  const { theme } = useTheme();
  const { language } = useLanguage();
  
  return (
    <div className="bg-card p-4 rounded-xl shadow-sm border border-border hover:shadow-md transition-shadow">
      <div className="flex items-start">
        <div className="w-10 h-10 rounded bg-blue-100 flex items-center justify-center text-blue-500 mr-3">
          {icon}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold mb-1">{title}</h3>
          <p className="text-sm text-muted-foreground mb-2">{description}</p>
          {isNew && (
            <span className="bg-blue-100 text-blue-600 text-xs px-2 py-1 rounded-full font-medium">
              {language === "zh" ? "新" : "NEW"}
            </span>
          )}
        </div>
      </div>
      <div className="flex justify-end mt-3">
        <Link 
          href={link}
          className="bg-muted hover:bg-muted/80 rounded px-3 py-1 text-sm font-medium transition-colors"
        >
          {language === "zh" ? "打开" : "Open"}
        </Link>
      </div>
    </div>
  );
}

export default function GenerateSection() {
  const { t, language } = useLanguage();
  const cards = useGenerateCardData();

  return (
    <div className="mb-12">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">{language === "zh" ? "生成" : "Generate"}</h2>
        <Link 
          href="/tools" 
          className="text-sm text-primary flex items-center"
        >
          {language === "zh" ? "显示全部" : "Show all"}
          <ChevronRight className="h-4 w-4 ml-1" />
        </Link>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {cards.map((card, index) => (
          <GenerateCard
            key={index}
            title={card.title}
            description={card.description}
            icon={card.icon}
            link={card.link}
            isNew={card.isNew}
          />
        ))}
      </div>
    </div>
  );
}
