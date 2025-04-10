'use client'

import React, { useState, useRef, useEffect } from 'react'
import { cn } from "@/lib/utils"; // 引入 cn 工具函数

interface LandingNavProps {
 onTryFree: () => void
}

export const LandingNav: React.FC<LandingNavProps> = ({ onTryFree }) => {
 const [activeIndex, setActiveIndex] = useState(3); // 默认"免费试用"按钮高亮
 const navItemsRef = useRef<(HTMLAnchorElement | null)[]>([]);
 const [highlightStyle, setHighlightStyle] = useState({
  left: 0,
  width: 0,
  height: 0,
  opacity: 0,
 });

 // 计算并设置高亮块的位置和尺寸
 const calculateHighlight = () => {
    const activeElement = navItemsRef.current[activeIndex];
    if (activeElement) {
        setHighlightStyle({
            left: activeElement.offsetLeft,
            width: activeElement.offsetWidth,
            height: activeElement.offsetHeight,
            opacity: 1,
        });
    }
  };

  // 初始计算和窗口大小变化时重新计算
  useEffect(() => {
    calculateHighlight(); // 初始计算

    window.addEventListener('resize', calculateHighlight);
    return () => {
      window.removeEventListener('resize', calculateHighlight);
    };
  }, [activeIndex]); // 依赖 activeIndex，当它变化时重新计算


 return (
  <nav id="navbar" className="flex justify-center items-center p-4 md:p-6 w-full transition-opacity duration-500 ease-out fade-in">
   {/* 导航栏容器 - 使用磨砂白色效果 */}
   <div className="glass-effect-white flex items-center justify-between px-6 py-3 rounded-full w-full max-w-4xl">
     {/* 左侧Logo */}
     <div className="text-xl md:text-2xl font-semibold text-gray-800">智译视界 ::</div>

     {/* 右侧导航项 */}
     <div className="relative inline-flex items-center gap-1 bg-gray-200/40 backdrop-blur-sm rounded-lg p-1">
      {/* 高亮背景块 - 使用蓝色 */}
      <div
       className="absolute transition-all duration-300 ease-in-out rounded-md bg-blue-600 pointer-events-none"
       style={{
        left: `${highlightStyle.left}px`,
        width: `${highlightStyle.width}px`,
        height: `${highlightStyle.height}px`,
        opacity: highlightStyle.opacity,
        zIndex: 0 // 确保在文字下方
       }}
      />

      {/* 导航项 - 所有导航项使用相同的圆角和样式 */}
      <a
       ref={(el) => { navItemsRef.current[0] = el; }}
       href="#"
       className={cn(
              "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
              activeIndex === 0 ? "text-white" : "text-gray-600 hover:bg-gray-200/30" // 高亮时文字白色，否则深灰色并有悬停效果
            )}
       onMouseEnter={() => setActiveIndex(0)}
      >
       项目案例
      </a>
      <a
       ref={(el) => { navItemsRef.current[1] = el; }}
       href="#"
       className={cn(
              "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
              activeIndex === 1 ? "text-white" : "text-gray-600 hover:bg-gray-200/30"
            )}
       onMouseEnter={() => setActiveIndex(1)}
      >
       功能特性
      </a>
      <a
       ref={(el) => { navItemsRef.current[2] = el; }}
       href="#"
       className={cn(
              "relative z-10 flex items-center px-3 py-1 rounded-md text-sm whitespace-nowrap cursor-pointer transition-colors duration-300 ease-in-out",
              activeIndex === 2 ? "text-white" : "text-gray-600 hover:bg-gray-200/30"
            )}
       onMouseEnter={() => setActiveIndex(2)}
      >
       关于我们
      </a>

      {/* 免费试用按钮 - 使用相同圆角 */}
      <a
       ref={(el) => { navItemsRef.current[3] = el; }}
       href="#"
       onClick={(e) => {
        e.preventDefault();
        onTryFree();
       }}
       id="free-trial-button-top"
       className={cn(
              "relative z-10 flex items-center px-4 py-1 rounded-md text-sm font-medium whitespace-nowrap cursor-pointer transition-all duration-300 ease-in-out gap-1",
               activeIndex === 3 ? "text-white" : "text-gray-600 hover:bg-gray-200/30" // 用一致的样式，高亮效果由背景块提供
             )}
       onMouseEnter={() => setActiveIndex(3)}
      >
       免费试用
       <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 7h10v10"/>
        <path d="M7 17 17 7"/>
       </svg>
      </a>
     </div>
   </div>
  </nav>
 )
}