import { useState, useEffect } from "react";
import { Route, Switch, useLocation, Link } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import Header from "@/components/header";
import PageTransition from "@/components/page-transition";
import { lazy, Suspense } from "react";
import { LanguageProvider } from "@/hooks/use-language";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { ThemeProvider } from "@/hooks/use-theme";
import { ProtectedRoute } from "./lib/protected-route";

// 页面导入
import Home from "@/pages/home";
import Pricing from "@/pages/pricing";
import NotFound from "@/pages/not-found";
// import LoginPage from "./pages/LoginPage"; // 削除済のためコメントアウト
// import SignupPage from "./pages/SignupPage"; // 削除済のためコメントアウト
import AuthPage from "@/pages/auth-page"; // 正しいパス @/pages/auth-page を使用

// 懒加载新添加的页面以提高性能
const AudioTranscription = lazy(() => import("./pages/audio-transcription"));
const TextToSpeech = lazy(() => import("./pages/text-to-speech"));
const VideoTranslation = lazy(() => import("./pages/video-translation"));

// AppContent コンポーネントを作成して useAuth を使用
const AppContent = () => {
  const [location] = useLocation();
  // const { user, signOut, loading } = useAuth(); // user, signOut, loading は Header 内で使用されるためここでは不要

  // const handleLogout = async () => { // handleLogout も Header 内で処理
  //   await signOut();
  // };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      
      {/* <nav className="p-4 flex justify-end space-x-4"> // この nav ブロックを削除
        {loading ? (
          <p>Loading...</p>
        ) : user ? (
          <>
            <span>{user.email}</span>
            <button onClick={handleLogout} className="text-blue-500 hover:underline">Logout</button>
          </>
        ) : (
          <>
            <Link href="/auth" className="text-blue-500 hover:underline">Login / Sign Up</Link>
          </>
        )}
      </nav> */}

      <main className="w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 overflow-hidden">
        <PageTransition location={location}>
          <Switch>
            <Route path="/" component={Home} />
            <Route path="/pricing" component={Pricing} />
            <Route path="/auth" component={AuthPage} />
            <ProtectedRoute 
              path="/audio-transcription" 
              component={() => (
                <Suspense fallback={<div className="flex items-center justify-center h-[60vh]">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
                </div>}>
                  <AudioTranscription />
                </Suspense>
              )} 
            />
            <ProtectedRoute 
              path="/text-to-speech" 
              component={() => (
                <Suspense fallback={<div className="flex items-center justify-center h-[60vh]">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
                </div>}>
                  <TextToSpeech />
                </Suspense>
              )} 
            />
            <ProtectedRoute 
              path="/video-translation" 
              component={() => (
                <Suspense fallback={<div className="flex items-center justify-center h-[60vh]">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
                </div>}>
                  <VideoTranslation />
                </Suspense>
              )} 
            />
            <Route component={NotFound} />
          </Switch>
        </PageTransition>
      </main>
      <Toaster />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ThemeProvider>
          <LanguageProvider>
            <AppContent />
          </LanguageProvider>
        </ThemeProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
