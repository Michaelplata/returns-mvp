-- Run this in your Supabase SQL editor (Database → SQL Editor → New query)

create table if not exists companies (
  id uuid default gen_random_uuid() primary key,
  name text not null unique,
  created_at timestamptz default now()
);

create table if not exists return_cases (
  id uuid default gen_random_uuid() primary key,
  company_id uuid references companies(id) on delete cascade not null,
  return_id text,
  sku text,
  category text,
  reason_code text,
  days_since_purchase integer,
  item_condition text,
  order_value numeric(10,2),
  customer_return_count integer,
  proof_of_purchase text,
  decision text,
  created_at timestamptz default now()
);

create table if not exists decisions (
  id uuid default gen_random_uuid() primary key,
  company_id uuid references companies(id) on delete cascade not null,
  sku text,
  category text,
  reason_code text,
  days_since_purchase integer,
  item_condition text,
  order_value numeric(10,2),
  customer_return_count integer,
  proof_of_purchase text,
  ai_recommended_decision text,
  ai_confidence text,
  ai_source text,
  human_decision text,
  overridden boolean,
  created_at timestamptz default now()
);

create index if not exists return_cases_company_id on return_cases(company_id);
create index if not exists decisions_company_id on decisions(company_id);
