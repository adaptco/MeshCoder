/* Generated from tools.yaml */

export interface SearchWebInput {
  query: string;
  top_k?: number;
}

export interface FetchPageInput {
  url: string;
}

export interface StoreStateInput {
  agent_id: string;
  state_data: any;
  namespace?: string;
}

export interface QueryVectorInput {
  query: string;
  top_k?: number;
  domain?: string;
}

export interface IngestCodebaseInput {
  root_path: string;
  domain?: string;
}

export interface ListEmailsInput {
  max_results?: number;
  query?: string;
}

export interface ReadEmailInput {
  email_id: string;
}

export interface SendEmailInput {
  to: string;
  subject: string;
  body: string;
}

export interface GenerateSpatialTensorInput {
  text_queries: array;
  domain?: string;
}

