import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY; // 对应您 .env 文件中的 SUPABASE_KEY (anon key)

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Supabase URL and Anon Key must be defined in VITE_ environment variables (.env file)');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey); 