convert $1'[0]' -coalesce \( $2'[0]' -coalesce \) \
          +append -channel A -evaluate set 0 +channel \
          $1 -coalesce -delete 0 \
          null: \( $2 -coalesce \) \
          -gravity East  -layers Composite $3
