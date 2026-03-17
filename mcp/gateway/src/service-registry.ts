import axios from "axios";

export interface ServiceConfig {
  baseUrl: string;
  name: string;
  tools: string[];
}

export class ServiceRegistry {
  private services: Map<string, ServiceConfig[]> = new Map();

  constructor(initialServices: Record<string, string[]>) {
    for (const [name, urls] of Object.entries(initialServices)) {
      urls.forEach(url => this.register(name, url));
    }
  }

  public register(toolName: string, baseUrl: string) {
    if (!this.services.has(toolName)) {
      this.services.set(toolName, []);
    }
    this.services.get(toolName)!.push({ name: toolName, baseUrl, tools: [toolName] });
  }

  public get(toolName: string): ServiceConfig[] {
    return this.services.get(toolName) || [];
  }

  public async discoverServices(discoveryUrls: string[]) {
    for (const url of discoveryUrls) {
      try {
          // This is a placeholder for actual MCP discovery
          // In practice, this would hit /.well-known/mcp-capabilities
          console.log(`Discovering services at ${url}...`);
      } catch (error) {
          console.error(`Failed to discover services at ${url}`, error);
      }
    }
  }
}
