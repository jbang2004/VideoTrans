import { cn } from "@/lib/utils";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";

interface VideoModelCardProps {
  name: string;
  description: string;
  duration: string;
  icon: React.ReactNode;
  features?: string[];
  isExpensiveModel?: boolean;
  bgColor?: string;
  textColor?: string;
}

export default function VideoModelCard({
  name,
  description,
  duration,
  icon,
  features = [],
  isExpensiveModel = false,
  bgColor = "bg-blue-100 dark:bg-blue-900/30",
  textColor = "text-blue-500 dark:text-blue-300"
}: VideoModelCardProps) {
  const { language } = useLanguage();
  const { theme } = useTheme();
  
  const expensiveModelText = language === "zh" ? "高级模型" : "Expensive model";
  
  // Adjust colors for dark mode
  const bgColorClass = bgColor.includes("dark:") ? bgColor : `${bgColor} dark:bg-opacity-20`;
  const textColorClass = textColor.includes("dark:") ? textColor : `${textColor} dark:text-opacity-90`;
  
  return (
    <div className="bg-card p-4 rounded-xl border border-border hover:shadow-md transition-shadow">
      <div className="flex items-center mb-2">
        <div className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center mr-2",
          bgColorClass
        )}>
          <div className={cn("h-5 w-5", textColorClass)}>
            {icon}
          </div>
        </div>
        <h4 className="font-medium">{name}</h4>
      </div>
      <p className="text-sm text-muted-foreground mb-2">{description}</p>
      <div className="flex flex-wrap gap-1">
        <div className="text-xs text-muted-foreground/70">{duration}</div>
        
        {features.map((feature, index) => (
          <div key={index} className="text-xs bg-muted/50 text-foreground/70 px-2 py-0.5 rounded">
            {feature}
          </div>
        ))}
        
        {isExpensiveModel && (
          <div className="text-xs bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded">
            {expensiveModelText}
          </div>
        )}
      </div>
    </div>
  );
}
