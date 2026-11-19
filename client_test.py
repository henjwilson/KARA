import socket
import threading

HOST = "127.0.0.1"
PORT = 5002


def receive_messages(conn, disconnected):
    """Receive messages and detect when the client disconnects."""

    buffer = b""

    while not disconnected.is_set():
        try:
            data = conn.recv(4096)

            # Empty bytes mean the client closed its connection
            if not data:
                print("\n[DISCONNECTED] Client closed connection")
                break

            # Accumulate incoming data
            buffer += data

            # Process newline-delimited messages
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                print(f"\n[RECEIVED] {line.decode('utf-8', errors='replace')}")

        except OSError as e:
            if not disconnected.is_set():
                print(f"\n[CONNECTION ERROR] {e}")
            break

    disconnected.set()


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Listening on {HOST}:{PORT}")

    try:
        while True:
            print("\nWaiting for client...")
            conn, addr = server.accept()
            print(f"[CONNECTED] {addr}")

            disconnected = threading.Event()

            # Background thread listens for incoming responses
            receiver = threading.Thread(
                target=receive_messages,
                args=(conn, disconnected),
                daemon=True
            )
            receiver.start()

            try:
                while not disconnected.is_set():
                    message = input("Send message: ")

                    if message.lower() == "quit":
                        return

                    if message.lower() == "disconnect":
                        break

                    if disconnected.is_set():
                        break

                    # Send plain text or JSON entered in terminal
                    # [100, 200, "red", 45]
                    conn.sendall((message + "\n").encode("utf-8"))
                    print("[SENT]")

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("\n[DISCONNECTED] Unable to send")

            finally:
                disconnected.set()
                conn.shutdown(socket.SHUT_RDWR) if conn.fileno() != -1 else None
                conn.close()
                receiver.join(timeout=1)
                print("[CLOSED] Waiting for reconnection...")

    except KeyboardInterrupt:
        print("\nStopping server...")

    finally:
        server.close()


if __name__ == "__main__":
    main()