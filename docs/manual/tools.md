# Herramientas

manual-id: tools

Propósito: documentar herramientas registradas, schemas conversacionales, confirmación, ejemplos y límites sin inventar herramientas.

## Criterio

`tool_name` y confirmación se validan contra `Bootstrap.build_tool_registry()`. Los argumentos se validan contra `ArgumentSchemaRegistry` cuando existe un intent conversacional asociado. Si una herramienta no tiene schema conversacional registrado, se documenta como `sin schema conversacional registrado`.

| tool_name | categoria | descripcion | argumentos | confirmacion | ejemplo | resultado | limitaciones |
|---|---|---|---|---|---|---|---|
| `calendar_list_events` | calendario | List Google Calendar events in a bounded time range (read-only). | req:time_min,time_max; opt:max_results | NO | `Lista eventos del calendario` | Eventos normalizados | Solo lectura, calendario principal, maximo 20 eventos |
| `desktop.activate_window` | escritorio | Activate an existing desktop window. | sin schema conversacional registrado | NO | Uso interno estructurado con handle/título validado | Ventana activada | Sin intent conversacional directo |
| `desktop.bring_window_to_front` | ventanas | Bring a window to the foreground. | sin schema conversacional registrado | NO | Uso interno con handle conocido | Ventana al frente | Requiere handle válido |
| `desktop.capture_screenshot` | escritorio | Capture the full screen as PNG. | sin schema conversacional registrado | NO | Uso interno con output_dir opcional | Ruta PNG | Crea archivo de captura |
| `desktop.clear_clipboard` | portapapeles | Clear the clipboard. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Portapapeles vaciado | No expuesto por selector conversacional |
| `desktop.clipboard_has_text` | portapapeles | Return whether the clipboard contains Unicode text. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Booleano | No lee contenido |
| `desktop.close_application` | procesos | Request normal close for one process by PID. | sin schema conversacional registrado | NO | Uso interno con pid | Solicitud de cierre | Puede fallar sin ventanas visibles |
| `desktop.close_window` | ventanas | Request closing a window. | sin schema conversacional registrado | NO | Uso interno con handle | Solicitud de cierre | Requiere handle válido |
| `desktop.copy_clipboard_text` | portapapeles | Copy Unicode text into the clipboard. | sin schema conversacional registrado | NO | Uso interno con text | Longitud copiada | No expuesto por selector conversacional |
| `desktop.double_click` | escritorio | Perform a double click at absolute screen coordinates. | sin schema conversacional registrado | NO | Uso interno con x,y | Doble clic | Coordenadas absolutas |
| `desktop.get_cursor_position` | escritorio | Return the current cursor position. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Tupla x,y | Depende del escritorio activo |
| `desktop.get_foreground_window` | ventanas | Return the current foreground window. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Datos de ventana | Solo ventana actual |
| `desktop.get_process` | procesos | Return process information by PID. | sin schema conversacional registrado | NO | Uso interno con pid | Datos de proceso o None | Requiere pid |
| `desktop.get_screen_size` | escritorio | Return the primary screen size. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Tupla ancho,alto | Pantalla primaria |
| `desktop.get_window_rect` | ventanas | Return a window rectangle. | sin schema conversacional registrado | NO | Uso interno con handle | Rectángulo | Requiere handle válido |
| `desktop.is_process_running` | procesos | Return whether a process matching a query is running. | sin schema conversacional registrado | NO | Uso interno con query | Booleano | Coincidencia por consulta |
| `desktop.left_click` | escritorio | Perform a left click at absolute screen coordinates. | sin schema conversacional registrado | NO | Uso interno con x,y | Clic realizado | Coordenadas absolutas |
| `desktop.list_processes` | procesos | List running processes matching a query. | sin schema conversacional registrado | NO | Uso interno con query | Lista de procesos | No expuesto por selector conversacional |
| `desktop.list_windows` | ventanas | List visible windows matching a title. | req:title | NO | `Lista ventanas llamadas Code` | Lista de ventanas visibles | Busca por título |
| `desktop.maximize_window` | ventanas | Maximize a window. | sin schema conversacional registrado | NO | Uso interno con handle | Ventana maximizada | Requiere handle válido |
| `desktop.minimize_window` | ventanas | Minimize a window. | sin schema conversacional registrado | NO | Uso interno con handle | Ventana minimizada | Requiere handle válido |
| `desktop.move_cursor` | escritorio | Move the cursor to absolute screen coordinates. | sin schema conversacional registrado | NO | Uso interno con x,y | Cursor movido | Coordenadas absolutas |
| `desktop.move_resize_window` | ventanas | Move and resize a window. | sin schema conversacional registrado | NO | Uso interno con handle,x,y,width,height | Ventana movida y redimensionada | Valida geometría |
| `desktop.move_window` | ventanas | Move a window preserving size. | sin schema conversacional registrado | NO | Uso interno con handle,x,y | Ventana movida | Requiere handle válido |
| `desktop.open_application` | escritorio | Open an installed desktop application. | req:application | NO | `Abre VS Code` | Aplicación abierta o ya abierta | Depende de Windows |
| `desktop.open_file` | escritorio | Open an existing file. | req:path; opt:application | NO | `Abre README.md` | Archivo abierto | Requiere archivo existente |
| `desktop.open_folder` | escritorio | Open an existing folder. | sin schema conversacional registrado | NO | Uso interno con path | Carpeta abierta | Requiere carpeta existente |
| `desktop.paste_clipboard` | portapapeles | Paste clipboard content into an existing target window. | sin schema conversacional registrado | NO | Uso interno con window_title | Contenido pegado | Requiere texto en portapapeles |
| `desktop.press_hotkey` | escritorio | Send a keyboard shortcut to an existing target window. | req:keys,window_title | SI | `Pulsa Ctrl+S en Visual Studio Code` | Atajo enviado | Requiere confirmación y ventana existente |
| `desktop.read_clipboard_text` | portapapeles | Read Unicode text from the clipboard. | sin schema conversacional registrado | NO | Uso interno sin argumentos | Texto o None | No expuesto por selector conversacional |
| `desktop.resize_window` | ventanas | Resize a window preserving position. | sin schema conversacional registrado | NO | Uso interno con handle,width,height | Ventana redimensionada | Valida dimensiones |
| `desktop.restore_window` | ventanas | Restore a window. | sin schema conversacional registrado | NO | Uso interno con handle | Ventana restaurada | Requiere handle válido |
| `desktop.right_click` | escritorio | Perform a right click at absolute coordinates. | sin schema conversacional registrado | NO | Uso interno con x,y | Clic derecho | Coordenadas absolutas |
| `desktop.save_file` | escritorio | Save the active file in an existing target window. | sin schema conversacional registrado | NO | Uso interno con window_title | Archivo guardado | Usa Ctrl+S |
| `desktop.scroll_vertical` | escritorio | Scroll vertically. | sin schema conversacional registrado | NO | Uso interno con direction | Scroll | Solo arriba/abajo |
| `desktop.terminate_process` | procesos | Terminate one process by PID. | sin schema conversacional registrado | NO | Uso interno con pid | Proceso terminado | Protege procesos críticos |
| `desktop.type_text` | escritorio | Type text into an existing target window. | req:text,window_title | SI | `Escribe hola en Bloc de notas` | Texto escrito | Requiere confirmación y ventana existente |
| `list_directory` | archivos | List files and directories. | opt:path | NO | `Lista la carpeta tools` | Lista de nombres | No incluye metadatos |
| `project_tree` | proyecto | Return all Python files inside a project. | opt:path | NO | `Muestra el árbol del proyecto` | Lista de archivos Python | Ignora carpetas técnicas |
| `read_file` | archivos | Read a UTF-8 text file. | req:path | NO | `Lee README.md` | Contenido del archivo | Requiere texto UTF-8 |
| `write_file` | archivos | Write a UTF-8 text file. | req:path,content | SI | `Escribe hola en prueba.txt` | Archivo actualizado | Requiere confirmación |

| `training.create_pdf` | documentos | Create and open a PDF from generated training content. | req:content; opt:output_dir | SI | Uso interno tras TrainingAgent | PDF creado y abierto | Requiere confirmación explícita |