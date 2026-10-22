import sys
import os
import time
import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

bedrock_agent = boto3.client('bedrock-agent', region_name=config.AWS_REGION)

EMBEDDING_MODEL_ARN = f"arn:aws:bedrock:{config.AWS_REGION}::foundation-model/amazon.titan-embed-text-v2:0"
POLICY_BUCKET_ARN = f"arn:aws:s3:::{config.POLICY_BUCKET}"

KBS = [
    {
        'key': 'RETURNS_KB_ID',
        'name': 'novamart-returns-policy-kb',
        'description': 'Knowledge Base for NovaMart Returns Policy',
        'index_name': config.RETURNS_VECTOR_INDEX,
        'prefix': 'policies/returns/',
    },
    {
        'key': 'SHIPPING_KB_ID',
        'name': 'novamart-shipping-policy-kb',
        'description': 'Knowledge Base for NovaMart Shipping Policy',
        'index_name': config.SHIPPING_VECTOR_INDEX,
        'prefix': 'policies/shipping/',
    },
    {
        'key': 'WARRANTY_KB_ID',
        'name': 'novamart-warranty-policy-kb',
        'description': 'Knowledge Base for NovaMart Warranty Policy',
        'index_name': config.WARRANTY_VECTOR_INDEX,
        'prefix': 'policies/warranty/',
    },
]

def get_or_create_kb(kb_info):
    existing = bedrock_agent.list_knowledge_bases().get('knowledgeBaseSummaries', [])
    for item in existing:
        if item['name'] == kb_info['name']:
            print(f"Knowledge Base {kb_info['name']} already exists: {item['knowledgeBaseId']}")
            return item['knowledgeBaseId']

    print(f"Creating Knowledge Base {kb_info['name']}...")
    response = bedrock_agent.create_knowledge_base(
        name=kb_info['name'],
        description=kb_info['description'],
        roleArn=config.AGENTCORE_ROLE_ARN,
        knowledgeBaseConfiguration={
            'type': 'VECTOR',
            'vectorKnowledgeBaseConfiguration': {
                'embeddingModelArn': EMBEDDING_MODEL_ARN,
            }
        },
        storageConfiguration={
            'type': 'S3_VECTORS',
            's3VectorsConfiguration': {
                'vectorBucketArn': config.VECTOR_STORE_BUCKET_ARN,
                'indexName': kb_info['index_name'],
            }
        }
    )
    kb_id = response['knowledgeBase']['knowledgeBaseId']
    print(f"  Created KB: {kb_id}")
    return kb_id

def wait_for_kb_active(kb_id):
    print(f"Waiting for KB {kb_id} to become ACTIVE...", end='', flush=True)
    while True:
        res = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
        status = res['knowledgeBase']['status']
        if status == 'ACTIVE':
            print(" ACTIVE.")
            break
        elif status == 'FAILED':
            raise RuntimeError(f"KB {kb_id} failed: {res['knowledgeBase'].get('failureReasons')}")
        time.sleep(5)
        print(".", end='', flush=True)

def setup_data_source_and_sync(kb_id, kb_info):
    data_sources = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id).get('dataSourceSummaries', [])
    ds_id = None
    for ds in data_sources:
        if ds['name'] == f"{kb_info['name']}-ds":
            ds_id = ds['dataSourceId']
            break

    if not ds_id:
        print(f"Creating Data Source for KB {kb_id}...")
        ds_res = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=f"{kb_info['name']}-ds",
            description=f"S3 Data source for {kb_info['name']}",
            dataSourceConfiguration={
                'type': 'S3',
                's3Configuration': {
                    'bucketArn': POLICY_BUCKET_ARN,
                    'inclusionPrefixes': [kb_info['prefix']],
                }
            }
        )
        ds_id = ds_res['dataSource']['dataSourceId']
        print(f"  Created Data Source: {ds_id}")

    # Start ingestion job
    print(f"Starting ingestion sync for {kb_id} (DS: {ds_id})...")
    job_res = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
    )
    job_id = job_res['ingestionJob']['ingestionJobId']
    print(f"  Ingestion Job ID: {job_id}")

    # Wait for ingestion to complete
    print(f"Waiting for ingestion job {job_id} to complete...", end='', flush=True)
    while True:
        job = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id
        )['ingestionJob']
        status = job['status']
        if status == 'COMPLETE':
            print(f" COMPLETE! (Documents: {job.get('statistics', {}).get('numberOfDocumentsScanned', 0)})")
            break
        elif status in ('FAILED', 'STOPPED'):
            raise RuntimeError(f"Ingestion job {job_id} ended with status {status}: {job.get('failureReasons')}")
        time.sleep(5)
        print(".", end='', flush=True)

def main():
    kb_ids = {}
    for kb_info in KBS:
        kb_id = get_or_create_kb(kb_info)
        wait_for_kb_active(kb_id)
        setup_data_source_and_sync(kb_id, kb_info)
        kb_ids[kb_info['key']] = kb_id

    # Update .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    updated = {}
    new_lines = []
    for line in lines:
        for key, val in kb_ids.items():
            if line.startswith(f"{key}="):
                line = f"{key}={val}\n"
                updated[key] = True
        new_lines.append(line)

    for key, val in kb_ids.items():
        if key not in updated:
            new_lines.append(f"{key}={val}\n")

    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("\n✅ All 3 Knowledge Bases created and synced successfully!")
    print(f"RETURNS_KB_ID={kb_ids['RETURNS_KB_ID']}")
    print(f"SHIPPING_KB_ID={kb_ids['SHIPPING_KB_ID']}")
    print(f"WARRANTY_KB_ID={kb_ids['WARRANTY_KB_ID']}")

if __name__ == '__main__':
    main()
