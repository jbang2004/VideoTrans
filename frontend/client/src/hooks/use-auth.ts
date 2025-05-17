import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabaseClient';
import { User, Session } from '@supabase/supabase-js';

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
}

export function useAuth(): AuthState {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    session: null,
    loading: true,
  });

  useEffect(() => {
    const getInitialSession = async () => {
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError) {
        console.error("Error getting initial session:", sessionError);
      }
      
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError) {
        console.error("Error getting initial user:", userError);
      }

      setAuthState({ user, session, loading: false });
    };

    getInitialSession();

    const { data: authListener } = supabase.auth.onAuthStateChange((event, session) => {
      const user = session?.user ?? null;
      setAuthState({ user, session, loading: false });
    });

    return () => {
      if (authListener && typeof authListener.subscription?.unsubscribe === 'function') {
        authListener.subscription.unsubscribe();
      }
    };
  }, []);

  return authState;
} 