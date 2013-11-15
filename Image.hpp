#pragma once
#include <stddef.h>
#include <stdint.h>

struct Image
{
    //
    Image( const uint16_t& width, const uint16_t& height )
        : m_width( width ), m_height( height ), m_size( width * height )
    {
        m_image = new uint32_t[ m_width * m_height ];
    }

    ~Image()
    {
        if( m_image )
            delete[] m_image;
    }

    const uint16_t& getWidth() const
    {
        return m_width;
    }

    const uint16_t& getHeight() const
    {
        return m_height;
    }

    const uint32_t& getSize() const
    {
        return m_size;
    }

    uint32_t* const pointer() const
    {
        return m_image;
    }

    void clearByWhite()
    {
        clear( 0xFFFFFFFFu );
    }

    void clear( const uint32_t& color )
    {
        for( size_t i = 0; i < m_size; i++ )
            m_image[ i ] = color;
    }

    void writePixel( const uint16_t& x, const uint16_t& y, const uint32_t& color = 0u )
    {
        m_image[ y * m_width + x ] = color;
    }

private:
    //
    uint16_t    m_width;
    uint16_t    m_height;
    uint32_t    m_size;
    uint32_t*   m_image;

    //
    Image()
        : m_width( 0 ), m_height( 0 ), m_size( 0 ), m_image( NULL )
    {

    }

};

//
int save_png( const char* file_name, const Image & image );
//
uint8_t ALPHA( uint32_t argb );
uint8_t RED( uint32_t argb );
uint8_t GREEN( uint32_t argb );
uint8_t BLUE( uint32_t argb );
//
typedef uint8_t (*COLOR_t)( uint32_t argb );
static const COLOR_t COLOR[4] = {ALPHA, RED, GREEN, BLUE};
