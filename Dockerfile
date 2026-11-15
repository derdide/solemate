# The app is a static site. This image is only a small web server for it.
FROM caddy:2-alpine AS caddy

FROM alpine:3
# The binary is copied on its own so it loses the cap_net_bind_service file capability
# the official image sets — with it, the container cannot start under no-new-privileges.
COPY --from=caddy /usr/bin/caddy /usr/bin/caddy
COPY Caddyfile /etc/caddy/Caddyfile

# Only what the browser needs
COPY index.html /srv/
COPY static /srv/static
COPY templates /srv/templates

RUN adduser -S -u 10001 -H app
USER 10001

EXPOSE 8000
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
