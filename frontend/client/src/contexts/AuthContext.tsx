import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { Session, User, AuthError, SignUpWithPasswordCredentials, SignInWithPasswordCredentials } from '@supabase/supabase-js';
import { supabase } from '../lib/supabaseClient';

interface AuthContextType {
  session: Session | null;
  user: User | null;
  loading: boolean;
  error: AuthError | null;
  signIn: (credentials: SignInWithPasswordCredentials) => Promise<{ error: AuthError | null }>;
  signUp: (credentials: SignUpWithPasswordCredentials) => Promise<{ error: AuthError | null }>;
  signOut: () => Promise<{ error: AuthError | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<AuthError | null>(null);

  useEffect(() => {
    setLoading(true);
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    const { data: authListener } = supabase.auth.onAuthStateChange(
      async (_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
        setError(null); // Clear previous errors on auth state change
      }
    );

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, []);

  const signIn = async (credentials: SignInWithPasswordCredentials) => {
    setLoading(true);
    setError(null);
    const { data, error } = await supabase.auth.signInWithPassword(credentials);
    setLoading(false);
    if (error) setError(error);
    // Session and user state will be updated by onAuthStateChange listener
    return { error };
  };

  const signUp = async (credentials: SignUpWithPasswordCredentials) => {
    setLoading(true);
    setError(null);
    // Supabase no longer takes arbitrary data in options for signUp in the same way for client.
    // User metadata should be handled separately, e.g., by creating a profile table.
    const { data, error } = await supabase.auth.signUp(credentials);
    setLoading(false);
    if (error) setError(error);
    // Session and user state will be updated by onAuthStateChange listener if auto-confirm or email confirm leads to sign-in
    return { error };
  };

  const signOut = async () => {
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.signOut();
    setLoading(false);
    if (error) setError(error);
    // Session and user state will be updated by onAuthStateChange listener
    return { error };
  };

  const value = {
    session,
    user,
    loading,
    error,
    signIn,
    signUp,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 