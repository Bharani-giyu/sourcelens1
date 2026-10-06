"""Offline regression tests; never import backend's persistent PDF store."""
import ast
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx


def load_client_functions():
    # Execute actual client code without creating Chroma or loading .env secrets.
    path = Path(__file__).resolve().parents[1] / 'backend.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    modules = {
        'os', 're', 'functools', 'uuid', 'pathlib', 'urllib.parse',
        'azure.ai.inference', 'azure.core.credentials', 'openai',
    }
    names = {
        'required', 'model_base_url', 'client', 'embedding_client', 'embed',
        'format_citations', 'pdf_path', 'delete_document',
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.Import) and all(a.name in modules for a in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module in modules)
        or (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'MAX_UPLOAD_BYTES'
            for target in node.targets
        ))
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class FoundryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.code = load_client_functions()
        self.host = 'https://example.services.ai.azure.com'
        self.code['client'].cache_clear()
        self.code['embedding_client'].cache_clear()

    def test_supported_urls_use_resource_route(self):
        for suffix in ('', '/', '/openai/v1', '/openai/v1/',
                       '/api/projects/demo', '/api/projects/demo/'):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    self.code['model_base_url'](self.host + suffix),
                    self.host + '/openai/v1/',
                )

    def test_invalid_urls_are_rejected(self):
        for url in ('http://example.com', 'https://user:secret@example.com',
                    self.host + '/api/projects/', self.host + '/models',
                    self.host + '?key=secret', self.host + '#fragment'):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                self.code['model_base_url'](url)

    def test_citations_map_to_retrieved_pdf_sources(self):
        answer, used = self.code['format_citations'](
            'Foundry IQ is a managed knowledge platform [S4].', 4,
        )
        self.assertEqual(
            answer,
            'Foundry IQ is a managed knowledge platform [4].',
        )
        self.assertEqual(used, [3])

    def test_out_of_range_citations_are_not_highlighted(self):
        answer, used = self.code['format_citations']('Claim [S4].', 3)
        self.assertEqual(used, [])
        self.assertIn('[invalid citation]', answer)
        self.assertIn('no PDF passages have been highlighted', answer)

    def test_upload_limit_cannot_exceed_20_mb(self):
        assignment = next(
            node for node in ast.parse(
                (Path(__file__).resolve().parents[1] / 'backend.py').read_text(
                    encoding='utf-8'
                )
            ).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == 'MAX_UPLOAD_BYTES'
                for target in node.targets
            )
        )
        with patch.dict(os.environ, {'MAX_UPLOAD_MB': '50'}):
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<test>', 'exec'), self.code)
        self.assertEqual(self.code['MAX_UPLOAD_BYTES'], 20 * 1024 * 1024)

    def test_delete_removes_pdf_manifest_and_vectors(self):
        class FakeCollection:
            def delete(self, *, where):
                self.where = where

        document_id = 'fe4eebc1-ff97-49ad-8329-97a2264593c4'
        collection = FakeCollection()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_dir = root / 'pdfs'
            manifest_dir = root / 'manifests'
            pdf_dir.mkdir()
            manifest_dir.mkdir()
            pdf_path = pdf_dir / f'{document_id}.pdf'
            manifest_path = manifest_dir / f'{document_id}.json'
            pdf_path.write_bytes(b'%PDF-test')
            manifest_path.write_text('{}', encoding='utf-8')

            self.code.update({
                'PDFS': pdf_dir,
                'MANIFESTS': manifest_dir,
                'STORE_LOCK': threading.RLock(),
                'collection': collection,
            })
            self.code['delete_document'](document_id)

            self.assertFalse(pdf_path.exists())
            self.assertFalse(manifest_path.exists())
            self.assertEqual(collection.where, {'document_id': document_id})

    def test_legacy_setting_and_foundry_precedence(self):
        project = self.host + '/api/projects/demo'
        for settings in (
              {'AZURE_OPENAI_ENDPOINT': project, 'AZURE_INFERENCE_API_KEY': 'test-key'},
            {'AZURE_AI_PROJECT_ENDPOINT': project,
               'AZURE_OPENAI_ENDPOINT': 'https://other.services.ai.azure.com',
               'AZURE_INFERENCE_API_KEY': 'test-key'},
        ):
            with self.subTest(settings=settings), patch.dict(os.environ, settings, clear=True):
                self.code['client'].cache_clear()
                with self.code['client']() as sdk:
                    self.assertEqual(str(sdk.base_url), self.host + '/openai/v1/')

    def test_chat_http_route_uses_foundry_api_key(self):
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.headers['authorization'], 'Bearer test-key')
            body = json.loads(request.content)
            self.assertEqual(request.url.path, '/openai/v1/chat/completions')
            self.assertEqual(body['model'], 'chat-test')
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'chat-test', 'choices': [{
                    'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'ok'},
                }],
            })

        settings = {
            'AZURE_AI_PROJECT_ENDPOINT': self.host + '/api/projects/demo',
            'AZURE_INFERENCE_API_KEY': 'test-key',
        }
        with patch.dict(os.environ, settings, clear=True):
            with self.code['client']() as sdk:
                with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
                    with sdk.with_options(http_client=transport) as mocked:
                        mocked.chat.completions.create(
                            model='chat-test', messages=[{'role': 'user', 'content': 'test'}],
                        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, '/openai/v1/chat/completions')

    def test_embeddings_use_foundry_key_endpoint(self):
        captured = {}
        batches = []
        test_key = 'test-key'

        class FakeEmbeddingsClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def embed(self, *, input):
                batches.append(input)
                return SimpleNamespace(data=[
                    SimpleNamespace(index=0, embedding=[0.1, 0.2]),
                ])

        settings = {
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': 'text-embedding-3-small',
            'AZURE_INFERENCE_ENDPOINT': self.host,
            'AZURE_INFERENCE_API_KEY': test_key,
        }
        with patch.dict(os.environ, settings, clear=True), patch.dict(
            self.code, {'EmbeddingsClient': FakeEmbeddingsClient},
        ):
            self.code['embedding_client'].cache_clear()
            self.assertEqual(self.code['embed'](['hello world']), [[0.1, 0.2]])

        self.assertEqual(
            captured['endpoint'],
            self.host + '/openai/deployments/text-embedding-3-small',
        )
        self.assertEqual(captured['credential'].key, test_key)
        self.assertEqual(
            captured['credential_scopes'],
            ['https://cognitiveservices.azure.com/.default'],
        )
        self.assertEqual(captured['api_version'], '2024-10-21')
        self.assertEqual(batches, [['hello world']])


if __name__ == '__main__':
    unittest.main()