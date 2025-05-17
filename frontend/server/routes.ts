import type { Express } from "express";
import { createServer, type Server } from "http";

export async function registerRoutes(app: Express): Promise<Server> {
  // Setup authentication routes
  // setupAuth(app); // Removed

  // Add any additional routes here
  // Always prefix with /api

  const httpServer = createServer(app);

  return httpServer;
}
