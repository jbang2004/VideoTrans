import { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import PricingToggle from "@/components/pricing-toggle";
import PlanCard, { PlanProps } from "@/components/plan-card";
import FaqAccordion from "@/components/faq-accordion";

export default function Pricing() {
  const [isYearly, setIsYearly] = useState(true);
  
  const handlePricingToggle = (yearly: boolean) => {
    setIsYearly(yearly);
  };

  const plans: Omit<PlanProps, 'onSelect'>[] = [
    {
      title: "Free",
      monthlyPrice: 0,
      features: [
        { title: "Free daily generations" },
        { title: "Limited access to KREA tools" }
      ],
      variant: "dark",
      cta: "Select Plan"
    },
    {
      title: "Basic",
      monthlyPrice: 10,
      discountedPrice: isYearly ? 8 : undefined,
      features: [
        { title: "~1,010 Flux images" },
        { title: "~36,000 real-time images" },
        { title: "~180 enhanced images" },
        { title: "~6 training jobs" },
        { title: "Commercial license" }
      ],
      variant: "green",
      cta: "Select Plan"
    },
    {
      title: "Pro",
      monthlyPrice: 35,
      discountedPrice: isYearly ? 28 : undefined,
      features: [
        { title: "~5,048 Flux images" },
        { title: "~180,000 real-time images" },
        { title: "~900 enhanced images" },
        { title: "~30 training jobs" },
        { title: "Commercial license" }
      ],
      variant: "blue",
      cta: "Select Plan"
    },
    {
      title: "Max",
      monthlyPrice: 60,
      discountedPrice: isYearly ? 48 : undefined,
      features: [
        { title: "~15,142 Flux images" },
        { title: "~540,000 real-time images" },
        { title: "~2,700 enhanced images" },
        { title: "~90 training jobs" },
        { title: "Commercial license" }
      ],
      variant: "purple",
      cta: "Select Plan"
    }
  ];

  const faqs = [
    {
      question: "What are compute units?",
      answer: "Compute units are a measure of computational resources used to generate images or videos. They represent the processing power, memory, and time required for each creation. Different tasks consume varying amounts of compute units based on their complexity and output quality."
    },
    {
      question: "Can I roll over unused compute units to the following month?",
      answer: "Compute units do not accumulate or carry over between billing cycles. At the start of each month, your compute unit balance is reset to your plan's allocated amount, regardless of any unused units from the previous month."
    },
    {
      question: "What options do I have if I exceed my compute unit limit?",
      answer: (
        <div>
          <p>If you exceed your compute unit limit, you have several options to continue your work:</p>
          <ol className="list-decimal pl-5 mt-2 space-y-1">
            <li>Upgrade to a higher-tier subscription with increased compute units.</li>
            <li>Purchase additional compute units to complement your current plan.</li>
            <li>Utilize your daily free compute units, which automatically refresh each day.</li>
          </ol>
        </div>
      )
    }
  ];

  const handleSelectPlan = (planTitle: string) => {
    console.log(`Selected plan: ${planTitle}`);
    // Handle plan selection logic here
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Krea Plans</h1>
        <p className="text-gray-600">Upgrade to gain access to Pro features and generate more, faster.</p>
        
        <div className="bg-blue-50 text-blue-600 py-2 px-4 rounded-lg inline-flex items-center mt-4 text-sm">
          <Sparkles className="h-5 w-5 mr-2" />
          Enterprise and team plans now available
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
        </div>
      </div>
      
      <PricingToggle onToggle={handlePricingToggle} defaultYearly={true} />
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        {plans.map((plan, index) => (
          <PlanCard
            key={index}
            {...plan}
            onSelect={() => handleSelectPlan(plan.title)}
          />
        ))}
      </div>
      
      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6 text-center">Frequently Asked Questions</h2>
        <FaqAccordion items={faqs} />
      </div>
      
      <div className="text-center text-sm text-gray-500">
        Need help with your subscription? Reach out to 
        <a href="mailto:support@krea.ai" className="text-blue-500 hover:underline ml-1">support@krea.ai</a>, 
        or chat directly with a team member on our discord.
      </div>
    </motion.div>
  );
}
