from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import quote
import secrets

from nicegui import app, events, run, ui
from starlette.responses import FileResponse, Response

import backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sourcelens')

ROOT = Path(__file__).resolve().parent
PDFJS = ROOT / 'static' / 'pdfjs'

if not (PDFJS / 'web' / 'viewer.html').exists():
    raise RuntimeError('Run "python setup_pdfjs.py" before starting the app.')

app.add_static_files('/pdfjs', str(PDFJS))


@app.get('/documents/{document_id}/original.pdf')
def original_pdf(document_id: str):
    try:
        path = backend.pdf_path(document_id)
    except ValueError:
        return Response(status_code=400)

    if not path.exists():
        return Response(status_code=404)

    return FileResponse(
        path,
        media_type='application/pdf',
        headers={
            'Content-Disposition': 'inline; filename="original.pdf"',
            'Cache-Control': 'no-store',
        },
    )


@app.get('/documents/{document_id}/highlighted.pdf')
@app.get('/documents/{document_id}/highlighted/{chunks}.pdf')
def annotated_pdf(document_id: str, chunks: str = ''):
    if len(chunks) > 5000:
        return Response('Request too long.', status_code=400)

    try:
        content = backend.highlighted_pdf(
            document_id,
            [item for item in chunks.split(',') if item],
        )
    except ValueError:
        return Response('Invalid highlight request.', status_code=400)
    except FileNotFoundError:
        return Response(status_code=404)
    except Exception:
        logger.exception('Could not create highlighted PDF.')
        return Response('Could not render PDF.', status_code=500)

    return Response(
        content,
        media_type='application/pdf',
        headers={
            'Content-Disposition': 'inline; filename="highlighted.pdf"',
            'Cache-Control': 'no-store',
        },
    )


STYLE = '''
<style>
body {
    background: #f3f5f8;
    color: #182335;
}
.nicegui-content {
    padding: 0 !important;
}
.workspace {
    display: grid;
    grid-template-columns: minmax(310px, 38%) minmax(0, 1fr);
    gap: 14px;
    height: calc(100vh - 100px);
}
.panel {
    background: white;
    border: 1px solid #dce2ea;
    border-radius: 12px;
    overflow: hidden;
}
.question {
    background: #182335;
    color: white;
    border-radius: 12px;
    padding: 12px 16px;
    white-space: pre-wrap;
}
.answer {
    font-size: 15px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}
.answer a {
    color: #075985;
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 2px;
}
.pdf-frame {
    border: 0;
    width: 100%;
    flex: 1;
    min-height: 0;
    background: #525659;
}
@media (max-width: 900px) {
    .workspace {
        grid-template-columns: 1fr;
        height: auto;
    }
    .chat-panel {
        height: 65vh;
    }
    .viewer-panel {
        height: 80vh;
    }
}
</style>
'''


class Workspace:
    def __init__(self) -> None:
        self.busy = False
        self.uploading = False
        self.upload_lock = asyncio.Lock()
        self.viewer_document: str | None = None

    def build(self) -> None:
        ui.add_head_html(STYLE)
        ui.colors(primary='#182335')

        with ui.column().classes('w-full p-4 gap-3'):
            with ui.row().classes(
                'w-full items-center justify-between'
            ):
                with ui.column().classes('gap-0'):
                    ui.label('SourceLens').classes(
                        'text-2xl font-bold'
                    )
                    ui.label(
                        'Ask questions. Check the evidence '
                        'in the original PDF.'
                    ).classes('text-sm text-gray-500')

                self.upload_button = ui.button(
                    'Upload PDF',
                    icon='upload_file',
                    on_click=lambda: self.upload_dialog.open(),
                ).props('unelevated no-caps')

            with ui.element('div').classes('workspace w-full'):
                self.build_chat()
                self.build_viewer()

        with ui.dialog() as self.upload_dialog:
            with ui.card().classes('w-96'):
                ui.label('Upload PDFs').classes(
                    'text-lg font-bold'
                )
                ui.label(
                    'Files stay on this machine. Extracted text '
                    'is sent to your configured Azure deployments.'
                ).classes('text-sm text-gray-500')

                ui.upload(
                    on_upload=self.upload,
                    auto_upload=True,
                    multiple=True,
                    max_file_size=backend.MAX_UPLOAD_BYTES,
                    on_rejected=lambda: ui.notify(
                        'Upload rejected. Choose a PDF no larger than 20 MB.',
                        type='warning',
                    ),
                ).props('accept=.pdf').classes('w-full')

                ui.button(
                    'Close',
                    on_click=self.upload_dialog.close,
                )

        self.refresh_documents()

    def build_chat(self) -> None:
        with ui.column().classes(
            'panel chat-panel h-full min-h-0 p-4 gap-3'
        ):
            self.selection = ui.select(
                options={'': 'All documents'},
                value='',
                label='Search within',
                on_change=self.select_document,
            ).props('outlined dense').classes('w-full')

            with ui.row().classes('w-full items-center justify-between'):
                ui.button(
                    'Refresh document list',
                    icon='refresh',
                    on_click=self.refresh_documents,
                ).props('flat dense no-caps')

                self.delete_button = ui.button(
                    'Delete PDF',
                    icon='delete',
                    on_click=self.prompt_delete,
                ).props('flat dense no-caps color=negative')
                self.delete_button.disable()

            with ui.row().classes('items-center gap-2 text-sm text-gray-600') as self.upload_status:
                ui.spinner(size='sm')
                self.upload_status_text = ui.label('Processing PDF…')
            self.upload_status.set_visibility(False)

            self.scroll = ui.scroll_area().classes(
                'w-full flex-1 min-h-0'
            )

            with self.scroll:
                self.messages = ui.column().classes(
                    'w-full gap-4 pr-3'
                )

                with self.messages:
                    ui.label(
                        'Upload a PDF, then ask a specific question. '
                        'Click an evidence button to jump to its page.'
                    ).classes('text-gray-500')

            self.question = ui.textarea(
                placeholder='Ask a question about the selected PDFs'
            ).props(
                'outlined autogrow input-style="max-height:140px"'
            ).classes('w-full')

            self.question.on(
                'keydown.enter.exact.prevent',
                self.ask,
            )

            self.send = ui.button(
                'Ask',
                icon='send',
                on_click=self.ask,
            ).props('unelevated no-caps').classes('self-end')

        with ui.dialog() as self.delete_dialog:
            with ui.card().classes('w-96'):
                ui.label('Delete uploaded PDF?').classes('text-lg font-bold')
                self.delete_confirmation = ui.label()
                with ui.row().classes('w-full justify-end gap-2'):
                    ui.button(
                        'Cancel', on_click=self.delete_dialog.close,
                    ).props('flat no-caps')
                    self.confirm_delete_button = ui.button(
                        'Delete', on_click=self.delete_selected,
                    ).props('unelevated no-caps color=negative')

    def build_viewer(self) -> None:
        with ui.column().classes(
            'panel viewer-panel h-full min-h-0 gap-0'
        ):
            with ui.row().classes(
                'w-full p-3 items-center justify-between'
            ):
                self.viewer_title = ui.label(
                    'Original PDF viewer'
                ).classes('font-semibold')

                self.original_button = ui.button(
                    'Show original',
                    on_click=self.show_original,
                ).props('flat dense no-caps')

                self.original_button.disable()

            self.viewer_container = ui.column().classes(
                'w-full flex-1 min-h-0 gap-0'
            )

            with self.viewer_container:
                self.frame = ui.element('iframe').props(
                    'title="Full PDF viewer"'
                ).classes('pdf-frame')

    def refresh_documents(self) -> None:
        documents = backend.list_documents()

        options = {'': 'All documents'}
        options.update({
            document['document_id']: (
                f"{document['file_name']} "
                f"({document['pages']} pages)"
            )
            for document in documents
        })

        value = self.selection.value
        if value not in options:
            value = ''

        self.selection.set_options(options, value=value)
        self.update_controls()

    def select_document(self, event) -> None:
        if event.value:
            self.open_pdf(event.value)
        else:
            self.clear_viewer()
        self.update_controls()

    def update_controls(self) -> None:
        processing = self.busy or self.uploading
        if processing:
            self.question.disable()
            self.send.disable()
            self.upload_button.disable()
        else:
            self.question.enable()
            self.send.enable()
            self.upload_button.enable()

        if self.selection.value and not processing:
            self.delete_button.enable()
        else:
            self.delete_button.disable()
        if processing:
            self.confirm_delete_button.disable()
        else:
            self.confirm_delete_button.enable()

    def clear_viewer(self) -> None:
        self.viewer_document = None
        self.viewer_container.clear()
        with self.viewer_container:
            self.frame = ui.element('iframe').props(
                'title="Full PDF viewer"'
            ).classes('pdf-frame')
        self.viewer_title.set_text('Original PDF viewer')
        self.original_button.disable()

    def prompt_delete(self) -> None:
        document_id = self.selection.value
        document = next(
            (item for item in backend.list_documents()
             if item['document_id'] == document_id),
            None,
        )
        if document is None:
            return
        self.delete_confirmation.set_text(
            f"'{document['file_name']}' and its indexed passages will be deleted."
        )
        self.delete_dialog.open()

    async def delete_selected(self) -> None:
        document_id = self.selection.value
        if not document_id:
            return
        self.busy = True
        self.update_controls()
        try:
            await run.io_bound(backend.delete_document, document_id)
        except Exception:
            logger.exception('Could not delete uploaded PDF.')
            ui.notify('Could not delete the PDF.', type='negative')
            return
        finally:
            self.busy = False
            self.update_controls()

        self.delete_dialog.close()
        self.refresh_documents()
        self.selection.set_value('')
        self.clear_viewer()
        ui.notify('PDF and indexed passages deleted.', type='positive')

    def open_pdf(
        self,
        document_id: str,
        page: int = 1,
        chunk_ids: list[str] | None = None,
    ) -> None:
        self.viewer_document = document_id
        page = max(1, int(page))

        if chunk_ids:
            file_url = (
                f'/documents/{document_id}/highlighted/'
                f'{",".join(chunk_ids)}.pdf'
            )
        else:
            file_url = (
                f'/documents/{document_id}/original.pdf'
            )

        viewer_url = (
            '/pdfjs/web/viewer.html'
            f'?file={quote(file_url, safe="")}'
            f'#page={page}&zoom=page-width'
        )

        # Recreate the iframe to apply the requested initial page.
        self.viewer_container.clear()

        with self.viewer_container:
            self.frame = ui.element('iframe').props(
                f'title="Full PDF viewer" src="{viewer_url}"'
            ).classes('pdf-frame')

        documents = backend.list_documents()

        name = next(
            (
                item['file_name']
                for item in documents
                if item['document_id'] == document_id
            ),
            'PDF',
        )

        mode = 'highlighted' if chunk_ids else 'original'

        self.viewer_title.set_text(
            f'{name} — page {page} · {mode}'
        )
        self.original_button.enable()

    def show_original(self) -> None:
        if self.viewer_document:
            self.open_pdf(self.viewer_document)

    async def upload(
        self,
        event: events.UploadEventArguments,
    ) -> None:
        async with self.upload_lock:
            self.uploading = True
            self.upload_status_text.set_text('Processing PDF…')
            self.upload_status.set_visibility(True)
            self.update_controls()
            try:
                # Support both newer and legacy upload event formats.
                uploaded_file = getattr(event, 'file', None)

                if uploaded_file is not None:
                    raw_name = uploaded_file.name
                else:
                    raw_name = event.name

                name = raw_name.replace('\\', '/').split('/')[-1]

                if not name.lower().endswith('.pdf'):
                    ui.notify(
                        'Please upload a PDF.',
                        type='warning',
                    )
                    return

                self.upload_status_text.set_text(f'Indexing {name}…')

                if uploaded_file is not None:
                    content = await uploaded_file.read()
                else:
                    event.content.seek(0)
                    content = await run.io_bound(
                        event.content.read
                    )

                document = await run.io_bound(
                    backend.ingest_pdf,
                    name,
                    content,
                )

            except ValueError as exc:
                ui.notify(str(exc), type='warning')
                return

            except Exception:
                logger.exception('PDF upload/indexing failed.')
                ui.notify(
                    'Indexing failed. Check the server log '
                    'and Azure settings.',
                    type='negative',
                )
                return

            finally:
                self.uploading = False
                self.upload_status.set_visibility(False)
                self.update_controls()

            self.refresh_documents()

            # Changing the selection invokes select_document,
            # which opens the uploaded PDF.
            self.selection.set_value(document['document_id'])

            ui.notify(
                f"Added {name}: {document['pages']} pages, "
                f"{document['passages']} passages.",
                type='positive',
            )

    def display_answer(
        self,
        answer: str,
        sources: list[dict],
        used: list[int],
    ) -> None:
        cited = [sources[index] for index in used]

        with self.messages:
            with ui.column().classes('w-full gap-2'):
                ui.markdown(answer).classes('answer')

                if cited:
                    ui.label('Supporting passages').classes(
                        'text-xs font-bold text-gray-500'
                    )

                    with ui.row().classes('gap-2'):
                        for index in used:
                            source = sources[index]

                            ids = [
                                item['chunk_id']
                                for item in cited
                                if item['document_id']
                                == source['document_id']
                            ]

                            button = ui.button(
                                f"[{index + 1}] "
                                f"{source['file_name']} "
                                f"· p. {source['page']}",
                                on_click=lambda _, s=source, c=ids: (
                                    self.open_pdf(
                                        s['document_id'],
                                        s['page'],
                                        c,
                                    )
                                ),
                            ).props(
                                'unelevated no-caps color=amber-2 text-color=black'
                            )

                            with button:
                                ui.tooltip(source['text'][:500])

        if cited:
            first = cited[0]

            self.open_pdf(
                first['document_id'],
                first['page'],
                [
                    item['chunk_id']
                    for item in cited
                    if item['document_id'] == first['document_id']
                ],
            )

    async def ask(self) -> None:
        question = (self.question.value or '').strip()

        if not question or self.busy:
            return

        if len(question) > 6000:
            ui.notify(
                'Please shorten your question.',
                type='warning',
            )
            return

        self.busy = True
        document_id = self.selection.value or None

        self.question.set_value('')
        self.update_controls()
        self.send.props('loading')

        with self.messages:
            ui.label(question).classes('question w-full')
            pending = ui.label(
                'Searching passages and generating an answer…'
            ).classes('text-gray-500')

        self.scroll.scroll_to(percent=1.0)

        try:
            answer, sources, used = await run.io_bound(
                backend.answer_question,
                question,
                document_id,
            )

            self.display_answer(answer, sources, used)

        except Exception:
            logger.exception('Answer generation failed.')

            with self.messages:
                ui.label(
                    'Could not generate an answer. '
                    'Check the server log and Azure configuration.'
                ).classes('text-red-600')

        finally:
            pending.delete()
            self.send.props(remove='loading')
            self.busy = False
            self.update_controls()
            self.scroll.scroll_to(percent=1.0)

@ui.page('/')
def index() -> None:
    Workspace().build()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='SourceLens',
        host=os.getenv('HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', '8080')),
        reload=False,
        show=False,
        favicon='🔎',
    )

    ui.run(
        title="SourceLens",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
        show=False,
    )
